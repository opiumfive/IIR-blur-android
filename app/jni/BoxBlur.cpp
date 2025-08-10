// Fast 2x/3x Box Blur (rolling-sum) with threading + NEON vertical (AArch64)
// - RGBA_8888, blur RGB, preserve A
// - Reflect-101 edges
// - Parallel H (rows) and V (column blocks)
// - Auto 2 boxes for sigma <= 8 (fewer passes), 3 boxes otherwise
//
// Build: -O3 -ffast-math -funroll-loops -std=c++17 (or C++14)
// Note: requires linking standard C++ threads (usually automatic in NDK).

#include <jni.h>
#include <android/bitmap.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

// ---------- misc helpers ----------

static inline int reflect101(int i, int n) {
    if (i < 0)  return -i;
    if (i >= n) return 2 * n - i - 2;
    return i;
}

// Generalized "boxes for gauss" (n=2 or n=3)
static inline void boxes_for_gauss(float sigma, int n, int* r) {
    if (sigma < 0.01f) { for (int i=0;i<n;++i) r[i]=0; return; }
    float wIdeal = std::sqrt((12.0f * sigma * sigma / n) + 1.0f);
    int wl = (int)std::floor(wIdeal);
    if ((wl & 1) == 0) wl--;
    int wu = wl + 2;
    float m_float = std::round((12.0f * sigma * sigma - n * wl * wl - 4.0f * n * wl - 3.0f * n)
                               / (-4.0f * wl - 4.0f));
    int m = (int)m_float;
    if (m < 0) m = 0;
    if (m > n) m = n;
    //int m = std::clamp((int)m_float, 0, n);
    for (int i=0; i<n; ++i) {
        int w = (i < m) ? wl : wu;
        r[i] = (w - 1) / 2;
    }
}

static inline unsigned clamp_threads(unsigned want) {
    if (want == 0) want = 4;
    return std::min(want, 8u); // avoid over-subscribing on mobile
}

// ---------- horizontal (row) pass: scalar but parallel ----------

static inline void hbox_row_rgba(const uint8_t* src, uint8_t* dst, int width, int radius)
{
    if (radius <= 0) { std::memcpy(dst, src, (size_t)width * 4); return; }
    const int w = 2*radius + 1;
    int rsum=0, gsum=0, bsum=0;

    // init at x=0
    for (int k = -radius; k <= radius; ++k) {
        int xx = reflect101(k, width);
        const uint8_t* p = src + ((size_t)xx<<2);
        rsum += p[0]; gsum += p[1]; bsum += p[2];
    }
    for (int x=0; x<width; ++x) {
        dst[((size_t)x<<2)+0] = (uint8_t)((rsum + w/2) / w);
        dst[((size_t)x<<2)+1] = (uint8_t)((gsum + w/2) / w);
        dst[((size_t)x<<2)+2] = (uint8_t)((bsum + w/2) / w);
        dst[((size_t)x<<2)+3] = src[((size_t)x<<2)+3];

        int xAdd = reflect101(x + radius + 1, width);
        int xSub = reflect101(x - radius,     width);
        const uint8_t* pAdd = src + ((size_t)xAdd<<2);
        const uint8_t* pSub = src + ((size_t)xSub<<2);
        rsum += (int)pAdd[0] - (int)pSub[0];
        gsum += (int)pAdd[1] - (int)pSub[1];
        bsum += (int)pAdd[2] - (int)pSub[2];
    }
}

static void hbox_pass_parallel(const uint8_t* srcBase, uint8_t* dstBase,
                               int width, int height, int stride, int radius,
                               unsigned threads)
{
    if (radius <= 0) {
        for (int y=0; y<height; ++y) {
            const uint8_t* src = srcBase + (size_t)y * (size_t)stride;
            uint8_t*       dst = dstBase + (size_t)y * (size_t)stride;
            std::memcpy(dst, src, (size_t)width * 4);
        }
        return;
    }
    threads = clamp_threads(threads);
    if (threads <= 1 || height < 2) {
        for (int y=0; y<height; ++y) {
            hbox_row_rgba(srcBase + (size_t)y * (size_t)stride,
                          dstBase + (size_t)y * (size_t)stride,
                          width, radius);
        }
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    int chunk = (height + (int)threads - 1) / (int)threads;
    for (unsigned t=0; t<threads; ++t) {
        int y0 = (int)t * chunk;
        int y1 = std::min(height, y0 + chunk);
        if (y0 >= y1) break;
        pool.emplace_back([=]() {
            for (int y = y0; y < y1; ++y) {
                const uint8_t* src = srcBase + (size_t)y * (size_t)stride;
                uint8_t*       dst = dstBase + (size_t)y * (size_t)stride;
                hbox_row_rgba(src, dst, width, radius);
            }
        });
    }
    for (auto& th : pool) th.join();
}

// ---------- vertical (column-block) pass: scalar + NEON variants ----------

static void vbox_block_scalar(const uint8_t* src, uint8_t* dst,
                              int width, int height, int stride, int radius,
                              int x0, int x1)
{
    const int w = 2*radius + 1;
    if (radius <= 0) {
        for (int y=0; y<height; ++y) {
            const uint8_t* s = src + (size_t)y * (size_t)stride + ((size_t)x0<<2);
            uint8_t*       d = dst + (size_t)y * (size_t)stride + ((size_t)x0<<2);
            std::memcpy(d, s, (size_t)(x1 - x0) * 4);
        }
        return;
    }

    int blk = x1 - x0;
    std::vector<uint32_t> sumR(blk), sumG(blk), sumB(blk);
    std::fill(sumR.begin(), sumR.end(), 0);
    std::fill(sumG.begin(), sumG.end(), 0);
    std::fill(sumB.begin(), sumB.end(), 0);

    // initial column sums at y=0
    for (int k=-radius; k<=radius; ++k) {
        int yy = reflect101(k, height);
        const uint8_t* row = src + (size_t)yy * (size_t)stride;
        for (int x=x0; x<x1; ++x) {
            const uint8_t* p = row + ((size_t)x<<2);
            int i = x - x0;
            sumR[i] += p[0]; sumG[i] += p[1]; sumB[i] += p[2];
        }
    }

    for (int y=0; y<height; ++y) {
        const uint8_t* rowC = src + (size_t)y * (size_t)stride; // alpha source
        uint8_t* out = dst + (size_t)y * (size_t)stride;

        for (int x=x0; x<x1; ++x) {
            int i = x - x0;
            out[((size_t)x<<2) + 0] = (uint8_t)((sumR[i] + w/2) / w);
            out[((size_t)x<<2) + 1] = (uint8_t)((sumG[i] + w/2) / w);
            out[((size_t)x<<2) + 2] = (uint8_t)((sumB[i] + w/2) / w);
            out[((size_t)x<<2) + 3] = rowC[((size_t)x<<2) + 3];
        }

        if (y+1 < height) {
            int yOut = reflect101(y - radius,     height);
            int yIn  = reflect101(y + radius + 1, height);
            const uint8_t* rowOut = src + (size_t)yOut * (size_t)stride;
            const uint8_t* rowIn  = src + (size_t)yIn  * (size_t)stride;
            for (int x=x0; x<x1; ++x) {
                const uint8_t* pOut = rowOut + ((size_t)x<<2);
                const uint8_t* pIn  = rowIn  + ((size_t)x<<2);
                int i = x - x0;
                sumR[i] += (int)pIn[0] - (int)pOut[0];
                sumG[i] += (int)pIn[1] - (int)pOut[1];
                sumB[i] += (int)pIn[2] - (int)pOut[2];
            }
        }
    }
}

#ifdef __aarch64__
// NEON block (fast path). Uses u16 accumulators, safe while window width <= 255.
static void vbox_block_neon(const uint8_t* src, uint8_t* dst,
                            int width, int height, int stride, int radius,
                            int x0, int x1)
{
    const int w = 2*radius + 1;
    if (radius <= 0) {
        for (int y=0; y<height; ++y) {
            const uint8_t* s = src + (size_t)y * (size_t)stride + ((size_t)x0<<2);
            uint8_t*       d = dst + (size_t)y * (size_t)stride + ((size_t)x0<<2);
            std::memcpy(d, s, (size_t)(x1 - x0) * 4);
        }
        return;
    }
    if (w > 255) { // avoid u16 overflow
        vbox_block_scalar(src, dst, width, height, stride, radius, x0, x1);
        return;
    }

    int blk = x1 - x0;
    std::vector<uint16_t> sumR(blk), sumG(blk), sumB(blk);
    std::fill(sumR.begin(), sumR.end(), 0);
    std::fill(sumG.begin(), sumG.end(), 0);
    std::fill(sumB.begin(), sumB.end(), 0);

    // Initial sums
    for (int k=-radius; k<=radius; ++k) {
        int yy = reflect101(k, height);
        const uint8_t* row = src + (size_t)yy * (size_t)stride;
        int x = x0;
        for (; x + 15 < x1; x += 16) {
            uint8x16x4_t v = vld4q_u8(row + ((size_t)x << 2));
            uint16x8_t r_lo = vmovl_u8(vget_low_u8 (v.val[0]));
            uint16x8_t r_hi = vmovl_u8(vget_high_u8(v.val[0]));
            uint16x8_t g_lo = vmovl_u8(vget_low_u8 (v.val[1]));
            uint16x8_t g_hi = vmovl_u8(vget_high_u8(v.val[1]));
            uint16x8_t b_lo = vmovl_u8(vget_low_u8 (v.val[2]));
            uint16x8_t b_hi = vmovl_u8(vget_high_u8(v.val[2]));

            uint16x8_t sr_lo = vld1q_u16(sumR.data() + (x - x0));
            uint16x8_t sr_hi = vld1q_u16(sumR.data() + (x - x0) + 8);
            uint16x8_t sg_lo = vld1q_u16(sumG.data() + (x - x0));
            uint16x8_t sg_hi = vld1q_u16(sumG.data() + (x - x0) + 8);
            uint16x8_t sb_lo = vld1q_u16(sumB.data() + (x - x0));
            uint16x8_t sb_hi = vld1q_u16(sumB.data() + (x - x0) + 8);

            sr_lo = vaddq_u16(sr_lo, r_lo); sr_hi = vaddq_u16(sr_hi, r_hi);
            sg_lo = vaddq_u16(sg_lo, g_lo); sg_hi = vaddq_u16(sg_hi, g_hi);
            sb_lo = vaddq_u16(sb_lo, b_lo); sb_hi = vaddq_u16(sb_hi, b_hi);

            vst1q_u16(sumR.data() + (x - x0),     sr_lo);
            vst1q_u16(sumR.data() + (x - x0) + 8, sr_hi);
            vst1q_u16(sumG.data() + (x - x0),     sg_lo);
            vst1q_u16(sumG.data() + (x - x0) + 8, sg_hi);
            vst1q_u16(sumB.data() + (x - x0),     sb_lo);
            vst1q_u16(sumB.data() + (x - x0) + 8, sb_hi);
        }
        for (; x < x1; ++x) {
            const uint8_t* p = row + ((size_t)x<<2);
            int i = x - x0;
            sumR[i] += p[0]; sumG[i] += p[1]; sumB[i] += p[2];
        }
    }

    // Fixed-point reciprocal (sum * scale + 0x8000) >> 16
    const uint16_t scale = (uint16_t)(((1u << 16) + (w/2)) / (unsigned)w);
    const uint16x8_t scale16 = vdupq_n_u16(scale);
    const uint32x4_t rnd32 = vdupq_n_u32(1u << 15);

    for (int y=0; y<height; ++y) {
        const uint8_t* rowC = src + (size_t)y * (size_t)stride;
        uint8_t* out = dst + (size_t)y * (size_t)stride;

        int x = x0;
        for (; x + 15 < x1; x += 16) {
            int i = x - x0;

            uint16x8_t sr_lo = vld1q_u16(sumR.data() + i);
            uint16x8_t sr_hi = vld1q_u16(sumR.data() + i + 8);
            uint16x8_t sg_lo = vld1q_u16(sumG.data() + i);
            uint16x8_t sg_hi = vld1q_u16(sumG.data() + i + 8);
            uint16x8_t sb_lo = vld1q_u16(sumB.data() + i);
            uint16x8_t sb_hi = vld1q_u16(sumB.data() + i + 8);

            // R
            uint32x4_t r0 = vmull_u16(vget_low_u16 (sr_lo), vget_low_u16 (scale16));
            uint32x4_t r1 = vmull_u16(vget_high_u16(sr_lo), vget_high_u16(scale16));
            uint32x4_t r2 = vmull_u16(vget_low_u16 (sr_hi), vget_low_u16 (scale16));
            uint32x4_t r3 = vmull_u16(vget_high_u16(sr_hi), vget_high_u16(scale16));
            r0 = vaddq_u32(r0, rnd32); r1 = vaddq_u32(r1, rnd32);
            r2 = vaddq_u32(r2, rnd32); r3 = vaddq_u32(r3, rnd32);
            r0 = vshrq_n_u32(r0, 16);  r1 = vshrq_n_u32(r1, 16);
            r2 = vshrq_n_u32(r2, 16);  r3 = vshrq_n_u32(r3, 16);
            uint16x8_t rAvg = vcombine_u16(vqmovn_u32(r0), vqmovn_u32(r1));
            uint16x8_t rAvg2= vcombine_u16(vqmovn_u32(r2), vqmovn_u32(r3));
            uint8x16_t r8   = vcombine_u8(vqmovn_u16(rAvg), vqmovn_u16(rAvg2));

            // G
            uint32x4_t g0 = vmull_u16(vget_low_u16 (sg_lo), vget_low_u16 (scale16));
            uint32x4_t g1 = vmull_u16(vget_high_u16(sg_lo), vget_high_u16(scale16));
            uint32x4_t g2 = vmull_u16(vget_low_u16 (sg_hi), vget_low_u16 (scale16));
            uint32x4_t g3 = vmull_u16(vget_high_u16(sg_hi), vget_high_u16(scale16));
            g0 = vaddq_u32(g0, rnd32); g1 = vaddq_u32(g1, rnd32);
            g2 = vaddq_u32(g2, rnd32); g3 = vaddq_u32(g3, rnd32);
            g0 = vshrq_n_u32(g0, 16);  g1 = vshrq_n_u32(g1, 16);
            g2 = vshrq_n_u32(g2, 16);  g3 = vshrq_n_u32(g3, 16);
            uint16x8_t gAvg = vcombine_u16(vqmovn_u32(g0), vqmovn_u32(g1));
            uint16x8_t gAvg2= vcombine_u16(vqmovn_u32(g2), vqmovn_u32(g3));
            uint8x16_t g8   = vcombine_u8(vqmovn_u16(gAvg), vqmovn_u16(gAvg2));

            // B
            uint32x4_t b0 = vmull_u16(vget_low_u16 (sb_lo), vget_low_u16 (scale16));
            uint32x4_t b1 = vmull_u16(vget_high_u16(sb_lo), vget_high_u16(scale16));
            uint32x4_t b2 = vmull_u16(vget_low_u16 (sb_hi), vget_low_u16 (scale16));
            uint32x4_t b3 = vmull_u16(vget_high_u16(sb_hi), vget_high_u16(scale16));
            b0 = vaddq_u32(b0, rnd32); b1 = vaddq_u32(b1, rnd32);
            b2 = vaddq_u32(b2, rnd32); b3 = vaddq_u32(b3, rnd32);
            b0 = vshrq_n_u32(b0, 16);  b1 = vshrq_n_u32(b1, 16);
            b2 = vshrq_n_u32(b2, 16);  b3 = vshrq_n_u32(b3, 16);
            uint16x8_t bAvg = vcombine_u16(vqmovn_u32(b0), vqmovn_u32(b1));
            uint16x8_t bAvg2= vcombine_u16(vqmovn_u32(b2), vqmovn_u32(b3));
            uint8x16_t b8   = vcombine_u8(vqmovn_u16(bAvg), vqmovn_u16(bAvg2));

            uint8x16x4_t aRow = vld4q_u8(rowC + ((size_t)x << 2));
            uint8x16x4_t out4;
            out4.val[0] = r8; out4.val[1] = g8; out4.val[2] = b8; out4.val[3] = aRow.val[3];
            vst4q_u8(out + ((size_t)x << 2), out4);
        }
        for (; x < x1; ++x) {
            int i = x - x0;
            out[((size_t)x<<2)+0] = (uint8_t)((sumR[i] + w/2) / w);
            out[((size_t)x<<2)+1] = (uint8_t)((sumG[i] + w/2) / w);
            out[((size_t)x<<2)+2] = (uint8_t)((sumB[i] + w/2) / w);
            out[((size_t)x<<2)+3] = rowC[((size_t)x<<2)+3];
        }

        if (y+1 < height) {
            int yOut = reflect101(y - radius,     height);
            int yIn  = reflect101(y + radius + 1, height);
            const uint8_t* rowOut = src + (size_t)yOut * (size_t)stride;
            const uint8_t* rowIn  = src + (size_t)yIn  * (size_t)stride;

            int x = x0;
            for (; x + 15 < x1; x += 16) {
                int i = x - x0;
                uint8x16x4_t vo = vld4q_u8(rowOut + ((size_t)x << 2));
                uint8x16x4_t vi = vld4q_u8(rowIn  + ((size_t)x << 2));

                uint16x8_t ro_lo = vmovl_u8(vget_low_u8 (vo.val[0]));
                uint16x8_t ro_hi = vmovl_u8(vget_high_u8(vo.val[0]));
                uint16x8_t ri_lo = vmovl_u8(vget_low_u8 (vi.val[0]));
                uint16x8_t ri_hi = vmovl_u8(vget_high_u8(vi.val[0]));
                uint16x8_t go_lo = vmovl_u8(vget_low_u8 (vo.val[1]));
                uint16x8_t go_hi = vmovl_u8(vget_high_u8(vo.val[1]));
                uint16x8_t gi_lo = vmovl_u8(vget_low_u8 (vi.val[1]));
                uint16x8_t gi_hi = vmovl_u8(vget_high_u8(vi.val[1]));
                uint16x8_t bo_lo = vmovl_u8(vget_low_u8 (vo.val[2]));
                uint16x8_t bo_hi = vmovl_u8(vget_high_u8(vo.val[2]));
                uint16x8_t bi_lo = vmovl_u8(vget_low_u8 (vi.val[2]));
                uint16x8_t bi_hi = vmovl_u8(vget_high_u8(vi.val[2]));

                uint16x8_t sr_lo = vld1q_u16(sumR.data() + i);
                uint16x8_t sr_hi = vld1q_u16(sumR.data() + i + 8);
                uint16x8_t sg_lo = vld1q_u16(sumG.data() + i);
                uint16x8_t sg_hi = vld1q_u16(sumG.data() + i + 8);
                uint16x8_t sb_lo = vld1q_u16(sumB.data() + i);
                uint16x8_t sb_hi = vld1q_u16(sumB.data() + i + 8);

                sr_lo = vaddq_u16(sr_lo, vsubq_u16(ri_lo, ro_lo));
                sr_hi = vaddq_u16(sr_hi, vsubq_u16(ri_hi, ro_hi));
                sg_lo = vaddq_u16(sg_lo, vsubq_u16(gi_lo, go_lo));
                sg_hi = vaddq_u16(sg_hi, vsubq_u16(gi_hi, go_hi));
                sb_lo = vaddq_u16(sb_lo, vsubq_u16(bi_lo, bo_lo));
                sb_hi = vaddq_u16(sb_hi, vsubq_u16(bi_hi, bo_hi));

                vst1q_u16(sumR.data() + i,     sr_lo); vst1q_u16(sumR.data() + i + 8, sr_hi);
                vst1q_u16(sumG.data() + i,     sg_lo); vst1q_u16(sumG.data() + i + 8, sg_hi);
                vst1q_u16(sumB.data() + i,     sb_lo); vst1q_u16(sumB.data() + i + 8, sb_hi);
            }
            for (; x < x1; ++x) {
                int i = x - x0;
                const uint8_t* pOut = rowOut + ((size_t)x<<2);
                const uint8_t* pIn  = rowIn  + ((size_t)x<<2);
                sumR[i] += (int)pIn[0] - (int)pOut[0];
                sumG[i] += (int)pIn[1] - (int)pOut[1];
                sumB[i] += (int)pIn[2] - (int)pOut[2];
            }
        }
    }
}
#endif

static void vbox_pass_parallel(const uint8_t* srcBase, uint8_t* dstBase,
                               int width, int height, int stride, int radius,
                               unsigned threads)
{
    threads = clamp_threads(threads);
    if (threads <= 1 || width < 64) {
#ifdef __aarch64__
        vbox_block_neon(srcBase, dstBase, width, height, stride, radius, 0, width);
#else
        vbox_block_scalar(srcBase, dstBase, width, height, stride, radius, 0, width);
#endif
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(threads);
    int chunk = (width + (int)threads - 1) / (int)threads;
    for (unsigned t=0; t<threads; ++t) {
        int x0 = (int)t * chunk;
        int x1 = std::min(width, x0 + chunk);
        if (x0 >= x1) break;
        pool.emplace_back([=]() {
#ifdef __aarch64__
            vbox_block_neon(srcBase, dstBase, width, height, stride, radius, x0, x1);
#else
            vbox_block_scalar(srcBase, dstBase, width, height, stride, radius, x0, x1);
#endif
        });
    }
    for (auto& th : pool) th.join();
}

// ---------- single box pass (H then V), parallel ----------

static inline void box_blur_single_pass(uint8_t* px, uint8_t* tmp,
                                        int width, int height, int stride, int radius,
                                        unsigned threads)
{
    // H: px -> tmp
    hbox_pass_parallel(px, tmp, width, height, stride, radius, threads);
    // V: tmp -> px
    vbox_pass_parallel(tmp, px, width, height, stride, radius, threads);
}

// ---------- public blur ----------

static void new_blur(uint8_t* px, int width, int height, int stride, float sigma)
{
    if (width <= 0 || height <= 0) return;

    // Heuristic: 2 boxes are often enough up to ~sigma 8 (UI blur), 3 otherwise
    int n = (sigma <= 8.0f ? 2 : 3);
    int r[3] = {0,0,0};
    boxes_for_gauss(sigma, n, r);

    unsigned threads = clamp_threads(std::thread::hardware_concurrency());

    uint8_t* tmp = (uint8_t*)std::malloc((size_t)height * (size_t)stride);
    if (!tmp) return;

    for (int i=0; i<n; ++i) {
        if (r[i] <= 0) continue;
        box_blur_single_pass(px, tmp, width, height, stride, r[i], threads);
    }

    std::free(tmp);
}

// ---------- JNI entry ----------

extern "C"
JNIEXPORT void JNICALL
Java_com_opiumfive_iirblurdemo_Utils_blurBox(JNIEnv* env, jclass,
                                             jobject bmp, jfloat sigma)
{
    AndroidBitmapInfo info;
    unsigned char* px;

    if (AndroidBitmap_getInfo(env, bmp, &info) != ANDROID_BITMAP_RESULT_SUCCESS)
        return;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return;

    if (AndroidBitmap_lockPixels(env, bmp, (void**)&px) != ANDROID_BITMAP_RESULT_SUCCESS)
        return;

    new_blur(px, (int)info.width, (int)info.height, (int)info.stride, (float)sigma);

    AndroidBitmap_unlockPixels(env, bmp);
}
