
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

#include <sched.h>
#include <pthread.h>

// TODO scalar branch seems to be broken

#ifdef __aarch64__

#include <arm_neon.h>

#endif

/* ----------------------------------------------------------- */
/* helpers for byte⇄float (point #3)                           */
/* ----------------------------------------------------------- */
#ifdef __aarch64__

static inline uint8x8_t f32_to_u8(float32x4_t lo, float32x4_t hi) {
    const float32x4_t half = vdupq_n_f32(0.5f);
    uint16x4_t lo16 = vmovn_u32(vcvtq_u32_f32(vaddq_f32(lo, half)));
    uint16x4_t hi16 = vmovn_u32(vcvtq_u32_f32(vaddq_f32(hi, half)));
    return vmovn_u16(vcombine_u16(lo16, hi16));
}

#endif

/* ----- load one column of four rows → three float32x4_t ----- */
#ifdef __aarch64__
#define LOAD_RGBA4(p0, p1, p2, p3, vR, vG, vB)                 \
    do {                                                       \
        vR = (float32x4_t){ (float)(p0)[0], (float)(p1)[0],    \
                            (float)(p2)[0], (float)(p3)[0] };  \
        vG = (float32x4_t){ (float)(p0)[1], (float)(p1)[1],    \
                            (float)(p2)[1], (float)(p3)[1] };  \
        vB = (float32x4_t){ (float)(p0)[2], (float)(p1)[2],    \
                            (float)(p2)[2], (float)(p3)[2] };  \
    } while (0)
#endif

/* =========================================================== */
/*                  minimal thread-pool                        */
/* =========================================================== */
struct ThreadPool {
    explicit ThreadPool(unsigned n) : done(false) {
        for (unsigned i = 0; i < n; ++i) {
            workers.emplace_back([this] { worker(); });
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
        cv_done.wait(lk, [this] { return queue.empty() && busy == 0; });
    }

private:
    void worker() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mx);
                cv.wait(lk, [this] { return done || !queue.empty(); });
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

    std::vector<std::thread> workers;
    std::vector<std::function<void()>> queue;
    std::mutex mx;
    std::condition_variable cv, cv_done;
    std::atomic<int> busy{0};
    bool done;
};

static ThreadPool &pool() {
    static ThreadPool p(std::max(2u, std::thread::hardware_concurrency()));
    return p;
}

/* =========================================================== */
/*  vertical SIMD pass                                         */
/* =========================================================== */
#ifdef __aarch64__

static inline void vert_pass_neon_4cols(float *bufBase,
                                        unsigned char *imgBase,
                                        unsigned columns,
                                        unsigned height,
                                        unsigned bufStride,
                                        unsigned imgStride,
                                        float b0, float b1, float b2, float B) {
    const float32x4_t vB = vdupq_n_f32(B);


    for (unsigned xb = 0; xb < columns * 3; xb += 12) {
        /* 1) bottom → top */
        float *ptr_buf = bufBase + xb + bufStride * (height - 1);

        float32x4x3_t s = vld3q_f32(ptr_buf);
        float32x4_t p0R = s.val[0], p1R = p0R, p2R = p0R;
        float32x4_t p0G = s.val[1], p1G = p0G, p2G = p0G;
        float32x4_t p0B = s.val[2], p1B = p0B, p2B = p0B;

        for (unsigned y = 0; y < height; ++y, ptr_buf -= bufStride) {

            /* pre-fetch next 4 rows to hide LPDDR latency */
            __builtin_prefetch(ptr_buf - bufStride * 4, /*rw=*/0, /*locality=*/1);
            s = vld3q_f32(ptr_buf);

            float32x4_t vR = vaddq_f32(vmulq_n_f32(s.val[0], vB[0]),vaddq_f32(vaddq_f32(vmulq_n_f32(p0R, b0),vmulq_n_f32(p1R, b1)),vmulq_n_f32(p2R, b2)));
            float32x4_t vG = vaddq_f32(vmulq_n_f32(s.val[1], vB[0]),vaddq_f32(vaddq_f32(vmulq_n_f32(p0G, b0),vmulq_n_f32(p1G, b1)),vmulq_n_f32(p2G, b2)));
            float32x4_t vBv = vaddq_f32(vmulq_n_f32(s.val[2], vB[0]),vaddq_f32(vaddq_f32(vmulq_n_f32(p0B, b0),vmulq_n_f32(p1B, b1)),vmulq_n_f32(p2B, b2)));

            float32x4x3_t tmp = {vR, vG, vBv};
            vst3q_f32(ptr_buf, tmp);

            p2R = p1R;
            p1R = p0R;
            p0R = vR;
            p2G = p1G;
            p1G = p0G;
            p0G = vG;
            p2B = p1B;
            p1B = p0B;
            p0B = vBv;
        }

        /* 2) top → bottom */
        ptr_buf = bufBase + xb;
        unsigned char *ptr_img = imgBase + (xb / 3) * 4;

        s = vld3q_f32(ptr_buf);
        p0R = p1R = p2R = s.val[0];
        p0G = p1G = p2G = s.val[1];
        p0B = p1B = p2B = s.val[2];

        for (unsigned y = 0; y < height; ++y, ptr_buf += bufStride, ptr_img += imgStride) {
            __builtin_prefetch(ptr_buf + bufStride * 4, 0, 1);
            s = vld3q_f32(ptr_buf);

            float32x4_t vR = vaddq_f32(vmulq_n_f32(s.val[0], vB[0]),vaddq_f32(vaddq_f32(vmulq_n_f32(p0R, b0),vmulq_n_f32(p1R, b1)),vmulq_n_f32(p2R, b2)));
            float32x4_t vG = vaddq_f32(vmulq_n_f32(s.val[1], vB[0]),vaddq_f32(vaddq_f32(vmulq_n_f32(p0G, b0),vmulq_n_f32(p1G, b1)),vmulq_n_f32(p2G, b2)));
            float32x4_t vBv = vaddq_f32(vmulq_n_f32(s.val[2], vB[0]),vaddq_f32(vaddq_f32(vmulq_n_f32(p0B, b0),vmulq_n_f32(p1B, b1)),vmulq_n_f32(p2B, b2)));

            p2R = p1R;
            p1R = p0R;
            p0R = vR;
            p2G = p1G;
            p1G = p0G;
            p0G = vG;
            p2B = p1B;
            p1B = p0B;
            p0B = vBv;

            uint8x8_t r8 = f32_to_u8(vR, vR);
            uint8x8_t g8 = f32_to_u8(vG, vG);
            uint8x8_t b8 = f32_to_u8(vBv, vBv);

            uint8x8x4_t out;
            out.val[0] = r8;
            out.val[1] = g8;
            out.val[2] = b8;
            out.val[3] = vdup_n_u8(255);

            vst4_lane_u8(ptr_img, out, 0);
            vst4_lane_u8(ptr_img + 4, out, 1);
            vst4_lane_u8(ptr_img + 8, out, 2);
            vst4_lane_u8(ptr_img + 12, out, 3);
        }
    }
}

#endif  /* __aarch64__ */

#define BLUR(dst, src, ch)  { dst = val##ch = B * (src) + (b0 * prev0##ch + b1 * prev1##ch + b2 * prev2##ch); \
                              prev2##ch = prev1##ch; prev1##ch = prev0##ch; prev0##ch = val##ch; }

#define BLUR_FINAL(dst, src, ch) { val##ch = B * (src) + (b0 * prev0##ch + b1 * prev1##ch + b2 * prev2##ch); \
                                   prev2##ch = prev1##ch; prev1##ch = prev0##ch; prev0##ch = val##ch; \
                                   dst = val##ch + 0.5f; }

#define SHARP(dst, src, ch) { val##ch = B * (src) + (b0 * prev0##ch + b1 * prev1##ch + b2 * prev2##ch); \
                              prev2##ch = prev1##ch; prev1##ch = prev0##ch; prev0##ch = val##ch; \
                              float sharpened##ch = (float)(dst) + ((float)(dst) - val##ch) * amount + 0.5f; \
                              dst = sharpened##ch < 0.f ? 0 : (sharpened##ch > 255.f ? 255 : sharpened##ch); }

/* =========================================================== */
/*          main separable IIR blur (SIMD horizontal)          */
/* =========================================================== */
static void iir_gauss_blur_u8_rgba_parallel(
        unsigned char *image,
        float *ext_buf,
        unsigned width,
        unsigned height,
        float sigma,
        float amount) {
    float q = sigma >= 2.5f
              ? 0.98711f * sigma - 0.96330f
              : 3.97156f - 4.14554f * sqrtf(1.f - 0.26891f * sigma);

    float d = 1.57825f + 2.44413f * q + 1.4281f * q * q + 0.422205f * q * q * q;
    float b0 = (2.44413f * q + 2.85619f * q * q + 1.26661f * q * q * q) / d;
    float b1 = -(1.4281f * q * q + 1.26661f * q * q * q) / d;
    float b2 = (0.422205f * q * q * q) / d;
    float B = 1.f - (b0 + b1 + b2);

    const unsigned imgStride = width * 4;
    const unsigned bufStride = width * 3;

    bool ownBuf = (ext_buf == nullptr);
    float *buffer = ownBuf ? nullptr : ext_buf;

    if (ownBuf && posix_memalign(reinterpret_cast<void **>(&buffer),
                                 64, sizeof(float) * bufStride * height))
        return;

#ifdef __aarch64__
    /* -------- 4-row SIMD horizontal pass -------- */
    const unsigned rowsPerTask = 128;

    for (unsigned y0 = 0; y0 < height; y0 += rowsPerTask) {
        unsigned y1 = std::min(y0 + rowsPerTask, height);

        pool().enqueue([=] {
            /* 4-row blocks */
            for (unsigned y = y0; y + 3 < y1; y += 4) {
                unsigned char *img0 = image + (y + 0) * imgStride;
                unsigned char *img1 = image + (y + 1) * imgStride;
                unsigned char *img2 = image + (y + 2) * imgStride;
                unsigned char *img3 = image + (y + 3) * imgStride;

                float *buf0 = buffer + (y + 0) * bufStride;
                float *buf1 = buffer + (y + 1) * bufStride;
                float *buf2 = buffer + (y + 2) * bufStride;
                float *buf3 = buffer + (y + 3) * bufStride;

                float32x4_t prev0R, prev0G, prev0B,
                        prev1R, prev1G, prev1B,
                        prev2R, prev2G, prev2B,
                        valR, valG, valB;

                LOAD_RGBA4(img0, img1, img2, img3, prev0R, prev0G, prev0B);
                prev1R = prev2R = prev0R;
                prev1G = prev2G = prev0G;
                prev1B = prev2B = prev0B;

                for (unsigned x = 0; x < width;
                     ++x,
                             img0 += 4, img1 += 4, img2 += 4, img3 += 4,
                             buf0 += 3, buf1 += 3, buf2 += 3, buf3 += 3) {
                    float32x4_t vR, vG, vB;
                    LOAD_RGBA4(img0, img1, img2, img3, vR, vG, vB);

                    /* val = B*src + b0*p0 + b1*p1 + b2*p2 (lane-wise) */
                    valR = vaddq_f32(vmulq_n_f32(vR, B),
                                     vaddq_f32(vaddq_f32(vmulq_n_f32(prev0R, b0),
                                                         vmulq_n_f32(prev1R, b1)),
                                               vmulq_n_f32(prev2R, b2)));
                    valG = vaddq_f32(vmulq_n_f32(vG, B),
                                     vaddq_f32(vaddq_f32(vmulq_n_f32(prev0G, b0),
                                                         vmulq_n_f32(prev1G, b1)),
                                               vmulq_n_f32(prev2G, b2)));
                    valB = vaddq_f32(vmulq_n_f32(vB, B),
                                     vaddq_f32(vaddq_f32(vmulq_n_f32(prev0B, b0),
                                                         vmulq_n_f32(prev1B, b1)),
                                               vmulq_n_f32(prev2B, b2)));

                    prev2R = prev1R;
                    prev1R = prev0R;
                    prev0R = valR;
                    prev2G = prev1G;
                    prev1G = prev0G;
                    prev0G = valG;
                    prev2B = prev1B;
                    prev1B = prev0B;
                    prev0B = valB;

                    float32x4x3_t s = {valR, valG, valB};
                    vst3q_lane_f32(buf0, s, 0);
                    vst3q_lane_f32(buf1, s, 1);
                    vst3q_lane_f32(buf2, s, 2);
                    vst3q_lane_f32(buf3, s, 3);
                }

                /* -------- reverse sweep (right → left), SIMD -------- */
                img0 -= 4;
                img1 -= 4;
                img2 -= 4;
                img3 -= 4;   /* restore ptrs */
                buf0 -= 3;
                buf1 -= 3;
                buf2 -= 3;
                buf3 -= 3;

                prev0R = prev1R = prev2R = valR;
                prev0G = prev1G = prev2G = valG;
                prev0B = prev1B = prev2B = valB;

                for (unsigned x = 0; x < width;
                     ++x,
                             img0 -= 4, img1 -= 4, img2 -= 4, img3 -= 4,
                             buf0 -= 3, buf1 -= 3, buf2 -= 3, buf3 -= 3) {
                    float32x4_t sR = (float32x4_t) {buf0[0], buf1[0], buf2[0], buf3[0]};
                    float32x4_t sG = (float32x4_t) {buf0[1], buf1[1], buf2[1], buf3[1]};
                    float32x4_t sB = (float32x4_t) {buf0[2], buf1[2], buf2[2], buf3[2]};

                    valR = vaddq_f32(vmulq_n_f32(sR, B),
                                     vaddq_f32(vaddq_f32(vmulq_n_f32(prev0R, b0),
                                                         vmulq_n_f32(prev1R, b1)),
                                               vmulq_n_f32(prev2R, b2)));
                    valG = vaddq_f32(vmulq_n_f32(sG, B),
                                     vaddq_f32(vaddq_f32(vmulq_n_f32(prev0G, b0),
                                                         vmulq_n_f32(prev1G, b1)),
                                               vmulq_n_f32(prev2G, b2)));
                    valB = vaddq_f32(vmulq_n_f32(sB, B),
                                     vaddq_f32(vaddq_f32(vmulq_n_f32(prev0B, b0),
                                                         vmulq_n_f32(prev1B, b1)),
                                               vmulq_n_f32(prev2B, b2)));

                    prev2R = prev1R;
                    prev1R = prev0R;
                    prev0R = valR;
                    prev2G = prev1G;
                    prev1G = prev0G;
                    prev0G = valG;
                    prev2B = prev1B;
                    prev1B = prev0B;
                    prev0B = valB;

                    buf0[0] = vgetq_lane_f32(valR, 0);
                    buf1[0] = vgetq_lane_f32(valR, 1);
                    buf2[0] = vgetq_lane_f32(valR, 2);
                    buf3[0] = vgetq_lane_f32(valR, 3);
                    buf0[1] = vgetq_lane_f32(valG, 0);
                    buf1[1] = vgetq_lane_f32(valG, 1);
                    buf2[1] = vgetq_lane_f32(valG, 2);
                    buf3[1] = vgetq_lane_f32(valG, 3);
                    buf0[2] = vgetq_lane_f32(valB, 0);
                    buf1[2] = vgetq_lane_f32(valB, 1);
                    buf2[2] = vgetq_lane_f32(valB, 2);
                    buf3[2] = vgetq_lane_f32(valB, 3);
                }
            }

            /* leftover rows <4 – scalar path */
            for (unsigned y = y0 + ((y1 - y0) & ~3u); y < y1; ++y) {
                unsigned char *ptr_img = image + y * imgStride;
                float *ptr_buf = buffer + y * bufStride;

                float prev00 = ptr_img[0], prev10 = ptr_img[0], prev20 = ptr_img[0];
                float prev01 = ptr_img[1], prev11 = ptr_img[1], prev21 = ptr_img[1];
                float prev02 = ptr_img[2], prev12 = ptr_img[2], prev22 = ptr_img[2];
                float val0{}, val1{}, val2{};

                for (unsigned x = 0; x < width; ++x, ptr_buf += 3, ptr_img += 4) {
                    BLUR(ptr_buf[0], ptr_img[0], 0);
                    BLUR(ptr_buf[1], ptr_img[1], 1);
                    BLUR(ptr_buf[2], ptr_img[2], 2);
                }
                ptr_buf -= 3;
                ptr_img -= 4;
                prev00 = prev10 = prev20 = val0;
                prev01 = prev11 = prev21 = val1;
                prev02 = prev12 = prev22 = val2;

                for (unsigned x = 0; x < width; ++x, ptr_buf -= 3, ptr_img -= 4) {
                    BLUR(ptr_buf[0], ptr_buf[0], 0);
                    BLUR(ptr_buf[1], ptr_buf[1], 1);
                    BLUR(ptr_buf[2], ptr_buf[2], 2);
                }
            }
        });
    }
    pool().wait_empty();
#else   /* -------- scalar horizontal (non-AArch64) -------- */
    for (unsigned y = 0; y < height; ++y) {
        unsigned char* ptr_img = image  + y * imgStride;
        float*         ptr_buf = buffer + y * bufStride;

        float prev00 = ptr_img[0], prev10 = ptr_img[0], prev20 = ptr_img[0];
        float prev01 = ptr_img[1], prev11 = ptr_img[1], prev21 = ptr_img[1];
        float prev02 = ptr_img[2], prev12 = ptr_img[2], prev22 = ptr_img[2];
        float val0{}, val1{}, val2{};

        for (unsigned x = 0; x < width; ++x, ptr_buf += 3, ptr_img += 4) {
            BLUR(ptr_buf[0], ptr_img[0], 0);
            BLUR(ptr_buf[1], ptr_img[1], 1);
            BLUR(ptr_buf[2], ptr_img[2], 2);
        }
        ptr_buf -= 3; ptr_img -= 4;
        prev00 = prev10 = prev20 = val0;
        prev01 = prev11 = prev21 = val1;
        prev02 = prev12 = prev22 = val2;

        for (unsigned x = 0; x < width; ++x, ptr_buf -= 3, ptr_img -= 4) {
            BLUR(ptr_buf[0], ptr_buf[0], 0);
            BLUR(ptr_buf[1], ptr_buf[1], 1);
            BLUR(ptr_buf[2], ptr_buf[2], 2);
        }
    }
#endif  /* __aarch64__ */

    /* ---------- vertical ---------- */
    const unsigned colsPerTask = 128;
    for (unsigned xb0 = 0; xb0 < bufStride; xb0 += colsPerTask * 3) {
        unsigned xb1 = std::min(xb0 + colsPerTask * 3, bufStride);
#ifdef __aarch64__
        unsigned colCount = (xb1 - xb0) / 3;
        unsigned col4 = colCount & ~3u;
        if (col4)
            pool().enqueue([=] {
                vert_pass_neon_4cols(buffer + xb0,
                                     image + (xb0 / 3) * 4,
                                     col4,
                                     height,
                                     bufStride, imgStride,
                                     b0, b1, b2, B);
            });
#endif  /* __aarch64__ */
        if (
#ifdef __aarch64__
                colCount != col4
#else
                true
#endif
                )
            pool().enqueue([=] {
                float prev00, prev01, prev02,
                        prev10, prev11, prev12,
                        prev20, prev21, prev22,
                        val0, val1, val2;
                for (unsigned xb = xb0 + (((xb1 - xb0) / 3 & ~3u) * 3);
                     xb < xb1; xb += 3) {
                    float *ptr_buf = buffer + xb + bufStride * (height - 1);
                    unsigned xPix = xb / 3;
                    unsigned char *ptr_img = image + xPix * 4;

                    prev00 = prev10 = prev20 = ptr_buf[0];
                    prev01 = prev11 = prev21 = ptr_buf[1];
                    prev02 = prev12 = prev22 = ptr_buf[2];
                    for (unsigned y = 0; y < height; ++y, ptr_buf -= bufStride) {
                        BLUR(ptr_buf[0], ptr_buf[0], 0);
                        BLUR(ptr_buf[1], ptr_buf[1], 1);
                        BLUR(ptr_buf[2], ptr_buf[2], 2);
                    }
                    ptr_buf += bufStride;
                    prev00 = prev10 = prev20 = val0;
                    prev01 = prev11 = prev21 = val1;
                    prev02 = prev12 = prev22 = val2;

                    if (amount > 0.f) {
                        for (unsigned y = 0; y < height; ++y,
                                ptr_buf += bufStride, ptr_img += imgStride) {
                            SHARP(ptr_img[0], ptr_buf[0], 0);
                            SHARP(ptr_img[1], ptr_buf[1], 1);
                            SHARP(ptr_img[2], ptr_buf[2], 2);
                        }
                    } else {
                        for (unsigned y = 0; y < height; ++y,
                                ptr_buf += bufStride, ptr_img += imgStride) {
                            BLUR_FINAL(ptr_img[0], ptr_buf[0], 0);
                            BLUR_FINAL(ptr_img[1], ptr_buf[1], 1);
                            BLUR_FINAL(ptr_img[2], ptr_buf[2], 2);
                        }
                    }
                }
            });
    }
    pool().wait_empty();

    if (ownBuf) free(buffer);
}

#undef BLUR
#undef BLUR_FINAL
#undef SHARP

extern "C"
JNIEXPORT void JNICALL
Java_com_opiumfive_iirblurdemo_Utils_blurNeon(JNIEnv *env, jclass, jobject bmp, jfloat sigma) {
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