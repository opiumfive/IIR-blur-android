#include <android/bitmap.h>
#include <jni.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <algorithm>

#include <unistd.h>
#include <pthread.h>

#ifdef __aarch64__
#include <arm_neon.h>
  #include <sys/auxv.h>
  #include <asm/hwcap.h>
#endif

class ThreadPool {
public:
    explicit ThreadPool(unsigned n) : done(false) {
        big_cores = detect_big_cores();
        if (!big_cores.empty()) {
            n = std::min<unsigned>(n, big_cores.size());
            n = std::max<unsigned>(1, n);
        } else {
            n = std::max(2u, n);
        }
        for (unsigned i = 0; i < n; ++i) {
            workers.emplace_back([this, i]{ worker(i); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mx);
            done = true;
            cv.notify_all();
        }
        for (auto &t: workers) t.join();
    }

    template<typename F>
    void enqueue(F &&f) {
        {
            std::lock_guard<std::mutex> lk(mx);
            queue.emplace_back(std::forward<F>(f));
        }
        cv.notify_one();
    }

    void wait_empty() {
        std::unique_lock<std::mutex> lk(mx);
        cv_done.wait(lk, [this]{ return queue.empty() && busy == 0; });
    }

private:
    void worker(unsigned idx) {
        // Pin each worker to a distinct big core (if we found any).
        pin_this_thread_to_big(idx);

        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mx);
                cv.wait(lk, [this]{ return done || !queue.empty(); });
                if (done && queue.empty()) return;
                job = std::move(queue.back());
                queue.pop_back();
                ++busy;
            }
            job();
            {
                std::lock_guard<std::mutex> lk(mx);
                --busy;
                if (queue.empty() && busy == 0) cv_done.notify_one();
            }
        }
    }

    // ---- Big core detection & pinning ---------------------------------
    static bool read_int_file(const char* path, int &out) {
        FILE* f = fopen(path, "r");
        if (!f) return false;
        char buf[64];
        size_t n = fread(buf, 1, sizeof(buf)-1, f);
        fclose(f);
        if (!n) return false;
        buf[n] = '\0';
        out = atoi(buf);
        return true;
    }

    static std::vector<int> detect_big_cores() {
        int ncpu = (int)sysconf(_SC_NPROCESSORS_CONF);
        if (ncpu <= 0) return {};
        struct Rec { int cpu; int freq; };
        std::vector<Rec> recs;
        recs.reserve(ncpu);

        for (int c = 0; c < ncpu; ++c) {
            char p1[128], p2[128];
            snprintf(p1, sizeof(p1), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
            snprintf(p2, sizeof(p2), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", c);
            int f = 0;
            if (!read_int_file(p1, f)) read_int_file(p2, f);
            recs.push_back({c, f});
        }

        std::sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b){
            return a.freq < b.freq; // ascending
        });

        // Take the top half as "big". Ensure at least 1 core.
        const int take = std::max(1, (int)recs.size()/2);
        std::vector<int> big;
        big.reserve(take);
        for (int i = (int)recs.size()-take; i < (int)recs.size(); ++i) {
            if (recs[i].freq > 0) big.push_back(recs[i].cpu);
        }

        // If we failed to read freqs (all zeros), assume last half are big.
        if (big.empty()) {
            for (int i = ncpu/2; i < ncpu; ++i) big.push_back(i);
        }
        return big;
    }

    void pin_this_thread_to_big(unsigned idx) {
        if (big_cores.empty()) return;
        const int cpu = big_cores[idx % big_cores.size()];
        cpu_set_t cs; CPU_ZERO(&cs);
        CPU_SET(cpu, &cs);
        (void)sched_setaffinity(pthread_self(), sizeof(cs), &cs);
    }

    std::vector<std::thread> workers;
    std::vector<std::function<void()>> queue;
    std::mutex mx;
    std::condition_variable cv, cv_done;
    std::atomic<int> busy{0};
    bool done;
    std::vector<int> big_cores;
};

static ThreadPool& pool() {
    static ThreadPool p(std::max(2u, std::thread::hardware_concurrency()));
    return p;
}

/* ===========================================================
 *                   Common utils / constants
 * =========================================================== */
#ifdef __aarch64__
static inline bool hw_has_fp16_accel() {
    unsigned long caps = getauxval(AT_HWCAP);
    return (caps & (HWCAP_FPHP | HWCAP_ASIMDHP)) != 0;
}

// Saturating f32 -> u8 pack for 4+4 lanes into a uint8x8_t
static inline uint8x8_t f32_to_u8_sat_8(float32x4_t lo, float32x4_t hi) {
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t z    = vdupq_n_f32(0.0f);
    const float32x4_t m255 = vdupq_n_f32(255.0f);
    lo = vminq_f32(vmaxq_f32(vaddq_f32(lo, half), z), m255);
    hi = vminq_f32(vmaxq_f32(vaddq_f32(hi, half), z), m255);
    uint16x4_t lo16 = vmovn_u32(vcvtq_u32_f32(lo));
    uint16x4_t hi16 = vmovn_u32(vcvtq_u32_f32(hi));
    // narrow to u8 (safe because we clamped to [0..255])
    return vmovn_u16(vcombine_u16(lo16, hi16));
}

// Pack 4 RGBA pixels (from r/g/b 4-lane vectors) into one 16B store.
static inline void store4_rgba_u8(uint8_t* dst, float32x4_t r, float32x4_t g, float32x4_t b) {
    uint8x8_t r8 = f32_to_u8_sat_8(r, r);
    uint8x8_t g8 = f32_to_u8_sat_8(g, g);
    uint8x8_t b8 = f32_to_u8_sat_8(b, b);
    uint8x8_t a8 = vdup_n_u8(255);

    // Interleave to RGBA and store 16 bytes
    uint8x8x2_t rg = vzip_u8(r8, g8);
    uint8x8x2_t ba = vzip_u8(b8, a8);
    uint8x16_t rgba = vreinterpretq_u8_u16(
        vzip1q_u16(vreinterpretq_u16_u8(vcombine_u8(rg.val[0], rg.val[1])),
                   vreinterpretq_u16_u8(vcombine_u8(ba.val[0], ba.val[1]))));
    vst1q_u8(dst, rgba);
}
#endif

/* Tunables */
static constexpr unsigned kRowsPerTask   = 96;   // 64–96 is a good range
static constexpr unsigned kColsPerTask   = 256;  // 192–256 tends to work well
static constexpr unsigned kPrefetchRows  = 3;    // 2–3 rows ahead

/* ===========================================================
 *     Main separable IIR Gaussian blur (planar FP16 scratch)
 * =========================================================== */

static void iir_gauss_blur_u8_rgba_parallel(
        unsigned char * __restrict image,  // RGBA, A left as-is or set to 255 (see store)
        float * __restrict ext_buf,        // ignored in FP16 path
        unsigned width,
        unsigned height,
        float sigma,
        float amount)                      // keep 0 for pure blur
{
    if (width == 0 || height == 0) return;

    // Coefficients (Deriche fit)
    float q = sigma >= 2.5f
              ? 0.98711f * sigma - 0.96330f
              : 3.97156f - 4.14554f * std::sqrt(1.f - 0.26891f * sigma);

    float d  = 1.57825f + 2.44413f * q + 1.4281f * q * q + 0.422205f * q * q * q;
    float b0 = (2.44413f * q + 2.85619f * q * q + 1.26661f * q * q * q) / d;
    float b1 = -(1.4281f  * q * q + 1.26661f * q * q * q) / d;
    float b2 =  (0.422205f * q * q * q) / d;
    float B  = 1.f - (b0 + b1 + b2);

    const unsigned imgStride = width * 4;

#ifdef __aarch64__
    // ---------------- Planar FP16 scratch path (preferred) ----------------
    const bool use_fp16 = (ext_buf == nullptr) && hw_has_fp16_accel();
    if (use_fp16)
    {
        const size_t planeElems = (size_t)width * height;
        __fp16* buf_all = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&buf_all), 64, sizeof(__fp16) * planeElems * 3))
            return;
        __fp16* planeR = buf_all;
        __fp16* planeG = buf_all + planeElems;
        __fp16* planeB = buf_all + planeElems * 2;

        // ---------------- Horizontal pass (4 rows/block), compute f32, store FP16 ----------------
        for (unsigned y0 = 0; y0 < height; y0 += kRowsPerTask) {
            const unsigned y1 = std::min(y0 + kRowsPerTask, height);
            pool().enqueue([=]() {
                // constants as vectors
                const float32x4_t vb0 = vdupq_n_f32(b0);
                const float32x4_t vb1 = vdupq_n_f32(b1);
                const float32x4_t vb2 = vdupq_n_f32(b2);
                const float32x4_t vB  = vdupq_n_f32(B);

                unsigned y = y0;
                for (; y + 3 < y1; y += 4) {
                    // Pointers for 4 rows
                    uint8_t* img0 = image + (y + 0) * imgStride;
                    uint8_t* img1 = image + (y + 1) * imgStride;
                    uint8_t* img2 = image + (y + 2) * imgStride;
                    uint8_t* img3 = image + (y + 3) * imgStride;

                    __fp16* r0 = planeR + (size_t)(y + 0) * width;
                    __fp16* r1 = planeR + (size_t)(y + 1) * width;
                    __fp16* r2 = planeR + (size_t)(y + 2) * width;
                    __fp16* r3 = planeR + (size_t)(y + 3) * width;

                    __fp16* g0 = planeG + (size_t)(y + 0) * width;
                    __fp16* g1 = planeG + (size_t)(y + 1) * width;
                    __fp16* g2 = planeG + (size_t)(y + 2) * width;
                    __fp16* g3 = planeG + (size_t)(y + 3) * width;

                    __fp16* b0p = planeB + (size_t)(y + 0) * width;
                    __fp16* b1p = planeB + (size_t)(y + 1) * width;
                    __fp16* b2p = planeB + (size_t)(y + 2) * width;
                    __fp16* b3p = planeB + (size_t)(y + 3) * width;

                    float32x4_t p0R, p0G, p0B;
                    float32x4_t p1R, p1G, p1B;
                    float32x4_t p2R, p2G, p2B;
                    float32x4_t vR, vG, vBv, valR, valG, valB;

                    // init previous (use first pixel of each row)
                    p0R = (float32x4_t){ (float)img0[0], (float)img1[0], (float)img2[0], (float)img3[0] };
                    p0G = (float32x4_t){ (float)img0[1], (float)img1[1], (float)img2[1], (float)img3[1] };
                    p0B = (float32x4_t){ (float)img0[2], (float)img1[2], (float)img2[2], (float)img3[2] };
                    p1R = p2R = p0R; p1G = p2G = p0G; p1B = p2B = p0B;

                    for (unsigned x = 0; x < width; ++x,
                         img0 += 4, img1 += 4, img2 += 4, img3 += 4) {
                        // load RGBA from 4 rows
                        vR = (float32x4_t){ (float)img0[0], (float)img1[0], (float)img2[0], (float)img3[0] };
                        vG = (float32x4_t){ (float)img0[1], (float)img1[1], (float)img2[1], (float)img3[1] };
                        vBv= (float32x4_t){ (float)img0[2], (float)img1[2], (float)img2[2], (float)img3[2] };

                        // val = B*src + b0*p0 + b1*p1 + b2*p2
                        valR = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(vR, vB), p0R, b0), p1R, b1), p2R, b2);
                        valG = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(vG, vB), p0G, b0), p1G, b1), p2G, b2);
                        valB = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(vBv, vB), p0B, b0), p1B, b1), p2B, b2);

                        p2R = p1R; p1R = p0R; p0R = valR;
                        p2G = p1G; p1G = p0G; p0G = valG;
                        p2B = p1B; p1B = p0B; p0B = valB;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                        float16x4_t r16 = vcvt_f16_f32(valR);
                        float16x4_t g16 = vcvt_f16_f32(valG);
                        float16x4_t b16 = vcvt_f16_f32(valB);
                        r0[x] = vget_lane_f16(r16,0); r1[x] = vget_lane_f16(r16,1);
                        r2[x] = vget_lane_f16(r16,2); r3[x] = vget_lane_f16(r16,3);
                        g0[x] = vget_lane_f16(g16,0); g1[x] = vget_lane_f16(g16,1);
                        g2[x] = vget_lane_f16(g16,2); g3[x] = vget_lane_f16(g16,3);
                        b0p[x]= vget_lane_f16(b16,0); b1p[x]= vget_lane_f16(b16,1);
                        b2p[x]= vget_lane_f16(b16,2); b3p[x]= vget_lane_f16(b16,3);
#else
                        r0[x] = (__fp16)vgetq_lane_f32(valR,0); r1[x] = (__fp16)vgetq_lane_f32(valR,1);
                        r2[x] = (__fp16)vgetq_lane_f32(valR,2); r3[x] = (__fp16)vgetq_lane_f32(valR,3);
                        g0[x] = (__fp16)vgetq_lane_f32(valG,0); g1[x] = (__fp16)vgetq_lane_f32(valG,1);
                        g2[x] = (__fp16)vgetq_lane_f32(valG,2); g3[x] = (__fp16)vgetq_lane_f32(valG,3);
                        b0p[x]= (__fp16)vgetq_lane_f32(valB,0); b1p[x]= (__fp16)vgetq_lane_f32(valB,1);
                        b2p[x]= (__fp16)vgetq_lane_f32(valB,2); b3p[x]= (__fp16)vgetq_lane_f32(valB,3);
#endif
                    }

                    // Reverse sweep (right->left), read FP16, write FP16 (planar)
                    // Restore pointers to end of each row
                    img0 -= 4; img1 -= 4; img2 -= 4; img3 -= 4;

                    // previous from last written values
                    float32x4_t pr = (float32x4_t){ (float)r0[width-1], (float)r1[width-1],
                                                    (float)r2[width-1], (float)r3[width-1] };
                    float32x4_t pg = (float32x4_t){ (float)g0[width-1], (float)g1[width-1],
                                                    (float)g2[width-1], (float)g3[width-1] };
                    float32x4_t pb = (float32x4_t){ (float)b0p[width-1], (float)b1p[width-1],
                                                    (float)b2p[width-1], (float)b3p[width-1] };
                    float32x4_t q1r = pr, q2r = pr, q1g = pg, q2g = pg, q1b = pb, q2b = pb;

                    for (unsigned xi = 0; xi < width; ++xi) {
                        unsigned x = width - 1 - xi;
                        float32x4_t sR = (float32x4_t){ (float)r0[x], (float)r1[x], (float)r2[x], (float)r3[x] };
                        float32x4_t sG = (float32x4_t){ (float)g0[x], (float)g1[x], (float)g2[x], (float)g3[x] };
                        float32x4_t sB = (float32x4_t){ (float)b0p[x],(float)b1p[x],(float)b2p[x],(float)b3p[x] };

                        float32x4_t vr = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sR, vB), pr, b0), q1r, b1), q2r, b2);
                        float32x4_t vg = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sG, vB), pg, b0), q1g, b1), q2g, b2);
                        float32x4_t vb = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sB, vB), pb, b0), q1b, b1), q2b, b2);

                        q2r = q1r; q1r = pr; pr = vr;
                        q2g = q1g; q1g = pg; pg = vg;
                        q2b = q1b; q1b = pb; pb = vb;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                        float16x4_t r16 = vcvt_f16_f32(vr);
                        float16x4_t g16 = vcvt_f16_f32(vg);
                        float16x4_t b16 = vcvt_f16_f32(vb);
                        r0[x] = vget_lane_f16(r16,0); r1[x] = vget_lane_f16(r16,1);
                        r2[x] = vget_lane_f16(r16,2); r3[x] = vget_lane_f16(r16,3);
                        g0[x] = vget_lane_f16(g16,0); g1[x] = vget_lane_f16(g16,1);
                        g2[x] = vget_lane_f16(g16,2); g3[x] = vget_lane_f16(g16,3);
                        b0p[x]= vget_lane_f16(b16,0); b1p[x]= vget_lane_f16(b16,1);
                        b2p[x]= vget_lane_f16(b16,2); b3p[x]= vget_lane_f16(b16,3);
#else
                        r0[x] = (__fp16)vgetq_lane_f32(vr,0); r1[x] = (__fp16)vgetq_lane_f32(vr,1);
                        r2[x] = (__fp16)vgetq_lane_f32(vr,2); r3[x] = (__fp16)vgetq_lane_f32(vr,3);
                        g0[x] = (__fp16)vgetq_lane_f32(vg,0); g1[x] = (__fp16)vgetq_lane_f32(vg,1);
                        g2[x] = (__fp16)vgetq_lane_f32(vg,2); g3[x] = (__fp16)vgetq_lane_f32(vg,3);
                        b0p[x]= (__fp16)vgetq_lane_f32(vb,0); b1p[x] = (__fp16)vgetq_lane_f32(vb,1);
                        b2p[x]= (__fp16)vgetq_lane_f32(vb,2); b3p[x] = (__fp16)vgetq_lane_f32(vb,3);
#endif
                    }
                }

                // leftover rows (<4): scalar store into planes
                for (; y < y1; ++y) {
                    uint8_t* img = image + y * imgStride;
                    __fp16* pr = planeR + (size_t)y * width;
                    __fp16* pg = planeG + (size_t)y * width;
                    __fp16* pb = planeB + (size_t)y * width;

                    float p00 = img[0], p10 = img[0], p20 = img[0];
                    float p01 = img[1], p11 = img[1], p21 = img[1];
                    float p02 = img[2], p12 = img[2], p22 = img[2];
                    float v0{}, v1{}, v2{};

                    for (unsigned x = 0; x < width; ++x, img += 4) {
                        v0 = B * img[0] + (b0 * p00 + b1 * p10 + b2 * p20);
                        v1 = B * img[1] + (b0 * p01 + b1 * p11 + b2 * p21);
                        v2 = B * img[2] + (b0 * p02 + b1 * p12 + b2 * p22);
                        p20 = p10; p10 = p00; p00 = v0;
                        p21 = p11; p11 = p01; p01 = v1;
                        p22 = p12; p12 = p02; p02 = v2;
                        pr[x] = (__fp16)v0; pg[x] = (__fp16)v1; pb[x] = (__fp16)v2;
                    }
                    // reverse
                    float rPrev = pr[width-1], r1Prev = rPrev, r2Prev = rPrev;
                    float gPrev = pg[width-1], g1Prev = gPrev, g2Prev = gPrev;
                    float bPrev = pb[width-1], b1Prev = bPrev, b2Prev = bPrev;

                    for (unsigned xi = 0; xi < width; ++xi) {
                        unsigned x = width - 1 - xi;
                        float s0 = (float)pr[x], s1 = (float)pg[x], s2 = (float)pb[x];
                        float vr = B * s0 + (b0 * rPrev + b1 * r1Prev + b2 * r2Prev);
                        float vg = B * s1 + (b0 * gPrev + b1 * g1Prev + b2 * g2Prev);
                        float vb = B * s2 + (b0 * bPrev + b1 * b1Prev + b2 * b2Prev);
                        r2Prev = r1Prev; r1Prev = rPrev; rPrev = vr;
                        g2Prev = g1Prev; g1Prev = gPrev; gPrev = vg;
                        b2Prev = b1Prev; b1Prev = bPrev; bPrev = vb;
                        pr[x] = (__fp16)vr; pg[x] = (__fp16)vg; pb[x] = (__fp16)vb;
                    }
                }
            });
        }
        pool().wait_empty();

        // ---------------- Vertical pass (process 8 columns at a time) ----------------
        auto vert_cols8 = [=](__fp16* rBase, __fp16* gBase, __fp16* bBase,
                              uint8_t* imgBase, unsigned xStart, unsigned colCount)
        {
            const float32x4_t vb0 = vdupq_n_f32(b0);
            const float32x4_t vb1 = vdupq_n_f32(b1);
            const float32x4_t vb2 = vdupq_n_f32(b2);
            const float32x4_t vB  = vdupq_n_f32(B);

            const size_t stride = width;

            for (unsigned x = xStart; x < xStart + colCount; x += 8) {
                // 1) bottom -> top: update planes with causal sweep
                __fp16* rptr = rBase + (size_t)(height - 1) * stride + x;
                __fp16* gptr = gBase + (size_t)(height - 1) * stride + x;
                __fp16* bptr = bBase + (size_t)(height - 1) * stride + x;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                float16x8_t r16 = vld1q_f16(rptr);
                float16x8_t g16 = vld1q_f16(gptr);
                float16x8_t b16 = vld1q_f16(bptr);
                float32x4_t p0R_lo = vcvt_f32_f16(vget_low_f16(r16));
                float32x4_t p0R_hi = vcvt_f32_f16(vget_high_f16(r16));
                float32x4_t p0G_lo = vcvt_f32_f16(vget_low_f16(g16));
                float32x4_t p0G_hi = vcvt_f32_f16(vget_high_f16(g16));
                float32x4_t p0B_lo = vcvt_f32_f16(vget_low_f16(b16));
                float32x4_t p0B_hi = vcvt_f32_f16(vget_high_f16(b16));
#else
                float32x4_t p0R_lo = { (float)rptr[0], (float)rptr[1], (float)rptr[2], (float)rptr[3] };
                float32x4_t p0R_hi = { (float)rptr[4], (float)rptr[5], (float)rptr[6], (float)rptr[7] };
                float32x4_t p0G_lo = { (float)gptr[0], (float)gptr[1], (float)gptr[2], (float)gptr[3] };
                float32x4_t p0G_hi = { (float)gptr[4], (float)gptr[5], (float)gptr[6], (float)gptr[7] };
                float32x4_t p0B_lo = { (float)bptr[0], (float)bptr[1], (float)bptr[2], (float)bptr[3] };
                float32x4_t p0B_hi = { (float)bptr[4], (float)bptr[5], (float)bptr[6], (float)bptr[7] };
#endif
                float32x4_t p1R_lo = p0R_lo, p2R_lo = p0R_lo;
                float32x4_t p1R_hi = p0R_hi, p2R_hi = p0R_hi;
                float32x4_t p1G_lo = p0G_lo, p2G_lo = p0G_lo;
                float32x4_t p1G_hi = p0G_hi, p2G_hi = p0G_hi;
                float32x4_t p1B_lo = p0B_lo, p2B_lo = p0B_lo;
                float32x4_t p1B_hi = p0B_hi, p2B_hi = p0B_hi;

                for (unsigned y = 0; y < height; ++y,
                     rptr -= stride, gptr -= stride, bptr -= stride) {

                    // Prefetch a few rows ahead
                    if (y + kPrefetchRows < height) {
                        __builtin_prefetch(rptr - (ptrdiff_t)stride * kPrefetchRows, 0, 1);
                        __builtin_prefetch(gptr - (ptrdiff_t)stride * kPrefetchRows, 0, 1);
                        __builtin_prefetch(bptr - (ptrdiff_t)stride * kPrefetchRows, 0, 1);
                    }

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                    float16x8_t r16r = vld1q_f16(rptr);
                    float16x8_t g16r = vld1q_f16(gptr);
                    float16x8_t b16r = vld1q_f16(bptr);
                    float32x4_t sR_lo = vcvt_f32_f16(vget_low_f16(r16r));
                    float32x4_t sR_hi = vcvt_f32_f16(vget_high_f16(r16r));
                    float32x4_t sG_lo = vcvt_f32_f16(vget_low_f16(g16r));
                    float32x4_t sG_hi = vcvt_f32_f16(vget_high_f16(g16r));
                    float32x4_t sB_lo = vcvt_f32_f16(vget_low_f16(b16r));
                    float32x4_t sB_hi = vcvt_f32_f16(vget_high_f16(b16r));
#else
                    float32x4_t sR_lo = { (float)rptr[0], (float)rptr[1], (float)rptr[2], (float)rptr[3] };
                    float32x4_t sR_hi = { (float)rptr[4], (float)rptr[5], (float)rptr[6], (float)rptr[7] };
                    float32x4_t sG_lo = { (float)gptr[0], (float)gptr[1], (float)gptr[2], (float)gptr[3] };
                    float32x4_t sG_hi = { (float)gptr[4], (float)gptr[5], (float)gptr[6], (float)gptr[7] };
                    float32x4_t sB_lo = { (float)bptr[0], (float)bptr[1], (float)bptr[2], (float)bptr[3] };
                    float32x4_t sB_hi = { (float)bptr[4], (float)bptr[5], (float)bptr[6], (float)bptr[7] };
#endif
                    float32x4_t vR_lo = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sR_lo, vB), p0R_lo, b0), p1R_lo, b1), p2R_lo, b2);
                    float32x4_t vR_hi = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sR_hi, vB), p0R_hi, b0), p1R_hi, b1), p2R_hi, b2);
                    float32x4_t vG_lo = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sG_lo, vB), p0G_lo, b0), p1G_lo, b1), p2G_lo, b2);
                    float32x4_t vG_hi = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sG_hi, vB), p0G_hi, b0), p1G_hi, b1), p2G_hi, b2);
                    float32x4_t vB_lo = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sB_lo, vB), p0B_lo, b0), p1B_lo, b1), p2B_lo, b2);
                    float32x4_t vB_hi = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sB_hi, vB), p0B_hi, b0), p1B_hi, b1), p2B_hi, b2);

                    // write back to FP16 planes
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                    vst1q_f16(rptr, vcombine_f16(vcvt_f16_f32(vR_lo), vcvt_f16_f32(vR_hi)));
                    vst1q_f16(gptr, vcombine_f16(vcvt_f16_f32(vG_lo), vcvt_f16_f32(vG_hi)));
                    vst1q_f16(bptr, vcombine_f16(vcvt_f16_f32(vB_lo), vcvt_f16_f32(vB_hi)));
#else
                    rptr[0] = (__fp16)vgetq_lane_f32(vR_lo,0);
                    rptr[1] = (__fp16)vgetq_lane_f32(vR_lo,1);
                    rptr[2] = (__fp16)vgetq_lane_f32(vR_lo,2);
                    rptr[3] = (__fp16)vgetq_lane_f32(vR_lo,3);
                    rptr[4] = (__fp16)vgetq_lane_f32(vR_hi,0);
                    rptr[5] = (__fp16)vgetq_lane_f32(vR_hi,1);
                    rptr[6] = (__fp16)vgetq_lane_f32(vR_hi,2);
                    rptr[7] = (__fp16)vgetq_lane_f32(vR_hi,3);

                    gptr[0] = (__fp16)vgetq_lane_f32(vG_lo,0);
                    gptr[1] = (__fp16)vgetq_lane_f32(vG_lo,1);
                    gptr[2] = (__fp16)vgetq_lane_f32(vG_lo,2);
                    gptr[3] = (__fp16)vgetq_lane_f32(vG_lo,3);
                    gptr[4] = (__fp16)vgetq_lane_f32(vG_hi,0);
                    gptr[5] = (__fp16)vgetq_lane_f32(vG_hi,1);
                    gptr[6] = (__fp16)vgetq_lane_f32(vG_hi,2);
                    gptr[7] = (__fp16)vgetq_lane_f32(vG_hi,3);

                    bptr[0] = (__fp16)vgetq_lane_f32(vB_lo,0);
                    bptr[1] = (__fp16)vgetq_lane_f32(vB_lo,1);
                    bptr[2] = (__fp16)vgetq_lane_f32(vB_lo,2);
                    bptr[3] = (__fp16)vgetq_lane_f32(vB_lo,3);
                    bptr[4] = (__fp16)vgetq_lane_f32(vB_hi,0);
                    bptr[5] = (__fp16)vgetq_lane_f32(vB_hi,1);
                    bptr[6] = (__fp16)vgetq_lane_f32(vB_hi,2);
                    bptr[7] = (__fp16)vgetq_lane_f32(vB_hi,3);
#endif
                    // advance state
                    p2R_lo = p1R_lo; p1R_lo = p0R_lo; p0R_lo = vR_lo;
                    p2R_hi = p1R_hi; p1R_hi = p0R_hi; p0R_hi = vR_hi;
                    p2G_lo = p1G_lo; p1G_lo = p0G_lo; p0G_lo = vG_lo;
                    p2G_hi = p1G_hi; p1G_hi = p0G_hi; p0G_hi = vG_hi;
                    p2B_lo = p1B_lo; p1B_lo = p0B_lo; p0B_lo = vB_lo;
                    p2B_hi = p1B_hi; p1B_hi = p0B_hi; p0B_hi = vB_hi;
                }

                // 2) top -> bottom: anti-causal and write out RGBA
                rptr = rBase + x;
                gptr = gBase + x;
                bptr = bBase + x;
                uint8_t* img = imgBase + x * 4;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                float16x8_t r16t = vld1q_f16(rptr);
                float16x8_t g16t = vld1q_f16(gptr);
                float16x8_t b16t = vld1q_f16(bptr);
                float32x4_t pR_lo = vcvt_f32_f16(vget_low_f16(r16t));
                float32x4_t pR_hi = vcvt_f32_f16(vget_high_f16(r16t));
                float32x4_t pG_lo = vcvt_f32_f16(vget_low_f16(g16t));
                float32x4_t pG_hi = vcvt_f32_f16(vget_high_f16(g16t));
                float32x4_t pB_lo = vcvt_f32_f16(vget_low_f16(b16t));
                float32x4_t pB_hi = vcvt_f32_f16(vget_high_f16(b16t));
#else
                float32x4_t pR_lo = { (float)rptr[0], (float)rptr[1], (float)rptr[2], (float)rptr[3] };
                float32x4_t pR_hi = { (float)rptr[4], (float)rptr[5], (float)rptr[6], (float)rptr[7] };
                float32x4_t pG_lo = { (float)gptr[0], (float)gptr[1], (float)gptr[2], (float)gptr[3] };
                float32x4_t pG_hi = { (float)gptr[4], (float)gptr[5], (float)gptr[6], (float)gptr[7] };
                float32x4_t pB_lo = { (float)bptr[0], (float)bptr[1], (float)bptr[2], (float)bptr[3] };
                float32x4_t pB_hi = { (float)bptr[4], (float)bptr[5], (float)bptr[6], (float)bptr[7] };
#endif
                float32x4_t q1R_lo = pR_lo, q2R_lo = pR_lo;
                float32x4_t q1R_hi = pR_hi, q2R_hi = pR_hi;
                float32x4_t q1G_lo = pG_lo, q2G_lo = pG_lo;
                float32x4_t q1G_hi = pG_hi, q2G_hi = pG_hi;
                float32x4_t q1B_lo = pB_lo, q2B_lo = pB_lo;
                float32x4_t q1B_hi = pB_hi, q2B_hi = pB_hi;

                for (unsigned y = 0; y < height; ++y,
                     rptr += stride, gptr += stride, bptr += stride, img += imgStride) {

                    if (y + kPrefetchRows < height) {
                        __builtin_prefetch(rptr + stride * kPrefetchRows, 0, 1);
                        __builtin_prefetch(gptr + stride * kPrefetchRows, 0, 1);
                        __builtin_prefetch(bptr + stride * kPrefetchRows, 0, 1);
                    }

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                    float16x8_t r16n = vld1q_f16(rptr);
                    float16x8_t g16n = vld1q_f16(gptr);
                    float16x8_t b16n = vld1q_f16(bptr);
                    float32x4_t sR_lo = vcvt_f32_f16(vget_low_f16(r16n));
                    float32x4_t sR_hi = vcvt_f32_f16(vget_high_f16(r16n));
                    float32x4_t sG_lo = vcvt_f32_f16(vget_low_f16(g16n));
                    float32x4_t sG_hi = vcvt_f32_f16(vget_high_f16(g16n));
                    float32x4_t sB_lo = vcvt_f32_f16(vget_low_f16(b16n));
                    float32x4_t sB_hi = vcvt_f32_f16(vget_high_f16(b16n));
#else
                    float32x4_t sR_lo = { (float)rptr[0], (float)rptr[1], (float)rptr[2], (float)rptr[3] };
                    float32x4_t sR_hi = { (float)rptr[4], (float)rptr[5], (float)rptr[6], (float)rptr[7] };
                    float32x4_t sG_lo = { (float)gptr[0], (float)gptr[1], (float)gptr[2], (float)gptr[3] };
                    float32x4_t sG_hi = { (float)gptr[4], (float)gptr[5], (float)gptr[6], (float)gptr[7] };
                    float32x4_t sB_lo = { (float)bptr[0], (float)bptr[1], (float)bptr[2], (float)bptr[3] };
                    float32x4_t sB_hi = { (float)bptr[4], (float)bptr[5], (float)bptr[6], (float)bptr[7] };
#endif
                    float32x4_t vR_lo = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sR_lo, vB), pR_lo, b0), q1R_lo, b1), q2R_lo, b2);
                    float32x4_t vR_hi = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sR_hi, vB), pR_hi, b0), q1R_hi, b1), q2R_hi, b2);
                    float32x4_t vG_lo = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sG_lo, vB), pG_lo, b0), q1G_lo, b1), q2G_lo, b2);
                    float32x4_t vG_hi = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sG_hi, vB), pG_hi, b0), q1G_hi, b1), q2G_hi, b2);
                    float32x4_t vB_lo = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sB_lo, vB), pB_lo, b0), q1B_lo, b1), q2B_lo, b2);
                    float32x4_t vB_hi = vfmaq_n_f32(vfmaq_n_f32(vfmaq_n_f32(vmulq_f32(sB_hi, vB), pB_hi, b0), q1B_hi, b1), q2B_hi, b2);

                    // Advance states
                    q2R_lo = q1R_lo; q1R_lo = pR_lo; pR_lo = vR_lo;
                    q2R_hi = q1R_hi; q1R_hi = pR_hi; pR_hi = vR_hi;
                    q2G_lo = q1G_lo; q1G_lo = pG_lo; pG_lo = vG_lo;
                    q2G_hi = q1G_hi; q1G_hi = pG_hi; pG_hi = vG_hi;
                    q2B_lo = q1B_lo; q1B_lo = pB_lo; pB_lo = vB_lo;
                    q2B_hi = q1B_hi; q1B_hi = pB_hi; pB_hi = vB_hi;

                    // Store: two 16-byte stores (4 pixels each)
                    store4_rgba_u8(img +  0, vR_lo, vG_lo, vB_lo);
                    store4_rgba_u8(img + 16, vR_hi, vG_hi, vB_hi);
                    // If you want A "untouched", copy it from original row instead of setting 255.
                }
            }
        };

        // Schedule vertical tasks
        for (unsigned xb0 = 0; xb0 < width; xb0 += kColsPerTask) {
            const unsigned cols = std::min(kColsPerTask, width - xb0);
            const unsigned cols8 = cols & ~7u;
            if (cols8) {
                pool().enqueue([=]() {
                    vert_cols8(planeR, planeG, planeB, image, xb0, cols8);
                });
            }
            // Tail columns (<8): scalar
            if (cols != cols8) {
                pool().enqueue([=]() {
                    for (unsigned x = xb0 + cols8; x < xb0 + cols; ++x) {
                        __fp16* rptr = planeR + (size_t)(height - 1) * width + x;
                        __fp16* gptr = planeG + (size_t)(height - 1) * width + x;
                        __fp16* bptr = planeB + (size_t)(height - 1) * width + x;

                        float p00 = (float)*rptr, p10 = p00, p20 = p00;
                        float p01 = (float)*gptr, p11 = p01, p21 = p01;
                        float p02 = (float)*bptr, p12 = p02, p22 = p02;

                        for (unsigned y = 0; y < height; ++y,
                             rptr -= width, gptr -= width, bptr -= width) {
                            float v0 = B * (float)*rptr + (b0 * p00 + b1 * p10 + b2 * p20);
                            float v1 = B * (float)*gptr + (b0 * p01 + b1 * p11 + b2 * p21);
                            float v2 = B * (float)*bptr + (b0 * p02 + b1 * p12 + b2 * p22);
                            *rptr = (__fp16)v0; *gptr = (__fp16)v1; *bptr = (__fp16)v2;
                            p20 = p10; p10 = p00; p00 = v0;
                            p21 = p11; p11 = p01; p01 = v1;
                            p22 = p12; p12 = p02; p02 = v2;
                        }

                        rptr += width; gptr += width; bptr += width;
                        uint8_t* img = image + x * 4;

                        p00 = (float)*rptr; p10 = p00; p20 = p00;
                        p01 = (float)*gptr; p11 = p01; p21 = p01;
                        p02 = (float)*bptr; p12 = p02; p22 = p02;

                        for (unsigned y = 0; y < height; ++y,
                             rptr += width, gptr += width, bptr += width, img += imgStride) {
                            float v0 = B * (float)*rptr + (b0 * p00 + b1 * p10 + b2 * p20);
                            float v1 = B * (float)*gptr + (b0 * p01 + b1 * p11 + b2 * p21);
                            float v2 = B * (float)*bptr + (b0 * p02 + b1 * p12 + b2 * p22);
                            p20 = p10; p10 = p00; p00 = v0;
                            p21 = p11; p11 = p01; p01 = v1;
                            p22 = p12; p12 = p02; p02 = v2;

                            // saturate to [0..255]
                            int R = (int)std::lrintf(std::max(0.f, std::min(255.f, v0)));
                            int G = (int)std::lrintf(std::max(0.f, std::min(255.f, v1)));
                            int Bc= (int)std::lrintf(std::max(0.f, std::min(255.f, v2)));
                            img[0] = (uint8_t)R;
                            img[1] = (uint8_t)G;
                            img[2] = (uint8_t)Bc;
                            img[3] = 255;
                        }
                    }
                });
            }
        }
        pool().wait_empty();

        free(buf_all);
        return;
    }
#endif // __aarch64__

    // ---------------- Fallback: FP32 scratch (portable scalar) ----------------
    const unsigned bufStride = width * 3;
    bool ownBuf = (ext_buf == nullptr);
    float* buffer = ownBuf ? nullptr : ext_buf;
    if (ownBuf && posix_memalign(reinterpret_cast<void**>(&buffer),
                                 64, sizeof(float) * bufStride * height))
        return;

    // Horizontal
    for (unsigned y = 0; y < height; ++y) {
        unsigned char* img = image  + y * imgStride;
        float*         buf = buffer + y * bufStride;

        float p00 = img[0], p10 = img[0], p20 = img[0];
        float p01 = img[1], p11 = img[1], p21 = img[1];
        float p02 = img[2], p12 = img[2], p22 = img[2];
        float v0{}, v1{}, v2{};

        for (unsigned x = 0; x < width; ++x, buf += 3, img += 4) {
            v0 = B * img[0] + (b0 * p00 + b1 * p10 + b2 * p20);
            v1 = B * img[1] + (b0 * p01 + b1 * p11 + b2 * p21);
            v2 = B * img[2] + (b0 * p02 + b1 * p12 + b2 * p22);
            p20 = p10; p10 = p00; p00 = v0;
            p21 = p11; p11 = p01; p01 = v1;
            p22 = p12; p12 = p02; p02 = v2;
            buf[0] = v0; buf[1] = v1; buf[2] = v2;
        }
        buf -= 3; img -= 4;
        p00 = p10 = p20 = v0; p01 = p11 = p21 = v1; p02 = p12 = p22 = v2;

        for (unsigned x = 0; x < width; ++x, buf -= 3, img -= 4) {
            v0 = B * buf[0] + (b0 * p00 + b1 * p10 + b2 * p20);
            v1 = B * buf[1] + (b0 * p01 + b1 * p11 + b2 * p21);
            v2 = B * buf[2] + (b0 * p02 + b1 * p12 + b2 * p22);
            p20 = p10; p10 = p00; p00 = v0;
            p21 = p11; p11 = p01; p01 = v1;
            p22 = p12; p12 = p02; p02 = v2;
            buf[0] = v0; buf[1] = v1; buf[2] = v2;
        }
    }

    // Vertical
    for (unsigned xb = 0; xb < bufStride; xb += 3) {
        float* ptr = buffer + xb + bufStride * (height - 1);
        unsigned xPix  = xb / 3;
        unsigned char* img = image  + xPix * 4;

        float p00 = ptr[0], p10 = p00, p20 = p00;
        float p01 = ptr[1], p11 = p01, p21 = p01;
        float p02 = ptr[2], p12 = p02, p22 = p02;

        for (unsigned y = 0; y < height; ++y, ptr -= bufStride) {
            float v0 = B * ptr[0] + (b0 * p00 + b1 * p10 + b2 * p20);
            float v1 = B * ptr[1] + (b0 * p01 + b1 * p11 + b2 * p21);
            float v2 = B * ptr[2] + (b0 * p02 + b1 * p12 + b2 * p22);
            ptr[0] = v0; ptr[1] = v1; ptr[2] = v2;
            p20 = p10; p10 = p00; p00 = v0;
            p21 = p11; p11 = p01; p01 = v1;
            p22 = p12; p12 = p02; p02 = v2;
        }
        ptr += bufStride;

        p00 = p10 = p20 = ptr[0];
        p01 = p11 = p21 = ptr[1];
        p02 = p12 = p22 = ptr[2];

        if (amount > 0.f) {
            for (unsigned y = 0; y < height; ++y, ptr += bufStride, img += imgStride) {
                float v0 = B * ptr[0] + (b0 * p00 + b1 * p10 + b2 * p20);
                float v1 = B * ptr[1] + (b0 * p01 + b1 * p11 + b2 * p21);
                float v2 = B * ptr[2] + (b0 * p02 + b1 * p12 + b2 * p22);
                p20 = p10; p10 = p00; p00 = v0;
                p21 = p11; p11 = p01; p01 = v1;
                p22 = p12; p12 = p02; p02 = v2;

                float s0 = img[0] + (img[0] - v0) * amount + 0.5f;
                float s1 = img[1] + (img[1] - v1) * amount + 0.5f;
                float s2 = img[2] + (img[2] - v2) * amount + 0.5f;
                img[0] = (unsigned char)(s0 < 0 ? 0 : (s0 > 255 ? 255 : s0));
                img[1] = (unsigned char)(s1 < 0 ? 0 : (s1 > 255 ? 255 : s1));
                img[2] = (unsigned char)(s2 < 0 ? 0 : (s2 > 255 ? 255 : s2));
            }
        } else {
            for (unsigned y = 0; y < height; ++y, ptr += bufStride, img += imgStride) {
                float v0 = B * ptr[0] + (b0 * p00 + b1 * p10 + b2 * p20);
                float v1 = B * ptr[1] + (b0 * p01 + b1 * p11 + b2 * p21);
                float v2 = B * ptr[2] + (b0 * p02 + b1 * p12 + b2 * p22);
                p20 = p10; p10 = p00; p00 = v0;
                p21 = p11; p11 = p01; p01 = v1;
                p22 = p12; p12 = p02; p02 = v2;

                int R = (int)(v0 + 0.5f);
                int G = (int)(v1 + 0.5f);
                int Bc= (int)(v2 + 0.5f);
                img[0] = (unsigned char)(R < 0 ? 0 : (R > 255 ? 255 : R));
                img[1] = (unsigned char)(G < 0 ? 0 : (G > 255 ? 255 : G));
                img[2] = (unsigned char)(Bc< 0 ? 0 : (Bc> 255 ? 255 : Bc));
            }
        }
    }

    if (ownBuf) free(buffer);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_opiumfive_iirblurdemo_Utils_blurNeonFp16(JNIEnv *env, jclass, jobject bmp, jfloat sigma) {
    AndroidBitmapInfo info;
    unsigned char *px = nullptr;
    if (AndroidBitmap_getInfo(env, bmp, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        AndroidBitmap_lockPixels(env, bmp, reinterpret_cast<void **>(&px))
        != ANDROID_BITMAP_RESULT_SUCCESS)
        return;

    iir_gauss_blur_u8_rgba_parallel(px, nullptr,
                                    info.width, info.height,
                                    sigma, 0.f);

    AndroidBitmap_unlockPixels(env, bmp);
}
