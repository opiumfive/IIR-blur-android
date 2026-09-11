#include "IIRBlur.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__aarch64__) && !defined(IIRBLUR_FORCE_SCALAR)
#include <arm_neon.h>
#endif

namespace iirblur {
namespace {
constexpr unsigned stripWidth = 4;
constexpr unsigned stripFloats = stripWidth * 3;
constexpr unsigned rowCount = 4;

// Reuses a bounded set of workers. A dispatch owns its completion barrier;
// overlapping callers can never release each other's scratch storage.
class Executor {
  public:
    Executor() {
        unsigned n = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
        workers.reserve(n - 1);
        for (unsigned i = 1; i < n; ++i) {
            try {
                workers.emplace_back([this] {
                    size_t seen = 0;
                    for (;;) {
                        std::unique_lock<std::mutex> lock(mutex);
                        wake.wait(lock, [&] { return stopping || generation != seen; });
                        if (stopping)
                            return;
                        seen = generation;
                        auto fn = execute;
                        void *arg = context;
                        lock.unlock();
                        fn(arg);
                        lock.lock();
                        if (--pending == 0)
                            done.notify_one();
                    }
                });
            } catch (const std::exception &) {
                break; // A device thread limit must not make the blur unusable.
            }
        }
    }
    ~Executor() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        wake.notify_all();
        for (auto &worker : workers)
            worker.join();
    }
    template <class F> void run(unsigned count, bool parallel, F &&f) {
        if (!parallel || count < 2 || workers.empty()) {
            for (unsigned i = 0; i < count; ++i)
                f(i);
            return;
        }
        std::lock_guard<std::mutex> dispatch(dispatchMutex);
        std::atomic<unsigned> next{0};
        std::exception_ptr error;
        auto drain = [&] {
            try {
                for (;;) {
                    unsigned i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= count)
                        break;
                    f(i);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!error)
                    error = std::current_exception();
                next.store(count, std::memory_order_relaxed);
            }
        };
        {
            std::lock_guard<std::mutex> lock(mutex);
            context = &drain;
            execute = [](void *arg) { (*static_cast<decltype(drain) *>(arg))(); };
            pending = workers.size();
            ++generation;
        }
        wake.notify_all();
        drain(); // The calling thread contributes work instead of sleeping.
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [&] { return pending == 0; });
        if (error)
            std::rethrow_exception(error);
    }

  private:
    std::vector<std::thread> workers;
    std::mutex mutex, dispatchMutex;
    std::condition_variable wake, done;
    void (*execute)(void *) = nullptr;
    void *context = nullptr;
    size_t generation = 0;
    unsigned pending = 0;
    bool stopping = false;
};

struct Coefficients {
    // Delta realization: acceleration, velocity, value. Avoid cancellation of
    // nearly equal feedback terms and preserve constant colours exactly.
    float velocity, acceleration, gain;
    explicit Coefficients(float sigma) {
        double q = sigma >= 2.5f ? 0.98711 * sigma - 0.96330
                                 : 3.97156 - 4.14554 * std::sqrt(1.0 - 0.26891 * sigma);
        q = std::max(0.0, q);
        double d = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
        acceleration = 0.422205 * q * q * q / d;
        gain = (1.57825 + 0.00001 * q * q) / d;
        velocity = (-1.57825 - 2.44413 * q - 0.000005 * q * q * q) / d;
    }
};

#if defined(__aarch64__) && !defined(IIRBLUR_FORCE_SCALAR)
struct State {
    float32x4_t value, velocity, acceleration;
    State() = default;
    explicit State(float32x4_t edge)
        : value(edge), velocity(vdupq_n_f32(0)), acceleration(vdupq_n_f32(0)) {}
    inline float32x4_t step(float32x4_t x, const Coefficients &k) {
        auto force = vfmaq_n_f32(vmulq_n_f32(acceleration, k.acceleration), velocity, k.velocity);
        acceleration = vfmaq_n_f32(force, vsubq_f32(x, value), k.gain);
        velocity = vaddq_f32(velocity, acceleration);
        value = vaddq_f32(value, velocity);
        return value;
    }
};

inline float32x4_t loadPixel(const uint8_t *p) {
    uint32_t packed;
    memcpy(&packed, p, 4);
    auto bytes = vreinterpret_u8_u32(vdup_n_u32(packed));
    return vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(bytes))));
}

void horizontal(uint8_t *pixels, float *buffer, unsigned w, unsigned h, size_t stride, unsigned y,
                const Coefficients k) {
    thread_local std::vector<float> scratch;
    scratch.resize(size_t(w) * rowCount * 4);
    float *rowScratch = scratch.data();
    unsigned rows = std::min(rowCount, h - y);
    const uint8_t *p[rowCount];
    State states[rowCount];
#pragma unroll
    for (unsigned r = 0; r < rowCount; ++r) {
        p[r] = pixels + (y + std::min(r, rows - 1)) * stride;
        states[r] = State(loadPixel(p[r]));
    }
    for (unsigned x = 0; x < w; ++x) {
        float *t = rowScratch + size_t(x) * rowCount * 4;
#pragma unroll
        for (unsigned r = 0; r < rowCount; ++r)
            vst1q_f32(t + r * 4, states[r].step(loadPixel(p[r] + x * 4), k));
    }
#pragma unroll
    for (unsigned r = 0; r < rowCount; ++r)
        states[r] = State(states[r].value);
    for (unsigned x = w; x-- > 0;) {
        const float *t = rowScratch + size_t(x) * rowCount * 4;
        float *dst = buffer + (size_t(x / stripWidth) * ((h + 3) & ~3u) + y) * stripFloats +
                     (x % stripWidth) * 3;
#pragma unroll
        for (unsigned r = 0; r < rowCount; ++r) {
            auto v = states[r].step(vld1q_f32(t + r * 4), k);
            if (r < rows) {
                vst1_f32(dst + r * stripFloats, vget_low_f32(v));
                vst1q_lane_f32(dst + r * stripFloats + 2, v, 2);
            }
        }
    }
}

void vertical(uint8_t *pixels, float *buffer, unsigned w, unsigned h, size_t stride, unsigned x,
              const Coefficients k) {
    float *base = buffer + size_t(x / stripWidth) * ((h + 3) & ~3u) * stripFloats;
    float *last = base + size_t(h - 1) * stripFloats;
    State states[stripFloats / 4];
#pragma unroll
    for (unsigned i = 0; i < stripFloats / 4; ++i)
        states[i] = State(vld1q_f32(last + i * 4));
    for (unsigned y = h; y-- > 0;) {
        float *t = base + size_t(y) * stripFloats;
#pragma unroll
        for (unsigned i = 0; i < stripFloats / 4; ++i)
            vst1q_f32(t + i * 4, states[i].step(vld1q_f32(t + i * 4), k));
    }
#pragma unroll
    for (unsigned i = 0; i < stripFloats / 4; ++i)
        states[i] = State(states[i].value);
    const uint8x16_t indices = {0, 1, 2, 16, 3, 4, 5, 16, 6, 7, 8, 16, 9, 10, 11, 16};
    const auto mask = vreinterpretq_u8_u32(vdupq_n_u32(0xff000000));
    for (unsigned y = 0; y < h; ++y) {
        const float *t = base + size_t(y) * stripFloats;
#pragma unroll
        for (unsigned i = 0; i < stripWidth / 4; ++i) {
            if (x + i * 4 >= w)
                continue;
            uint8_t *dst = pixels + y * stride + size_t(x + i * 4) * 4;
            auto v0 = states[i * 3].step(vld1q_f32(t + i * 12), k);
            auto v1 = states[i * 3 + 1].step(vld1q_f32(t + i * 12 + 4), k);
            auto v2 = states[i * 3 + 2].step(vld1q_f32(t + i * 12 + 8), k);
            auto a = vqmovn_u32(vcvtq_u32_f32(vaddq_f32(v0, vdupq_n_f32(.5f))));
            auto b = vqmovn_u32(vcvtq_u32_f32(vaddq_f32(v1, vdupq_n_f32(.5f))));
            auto c = vqmovn_u32(vcvtq_u32_f32(vaddq_f32(v2, vdupq_n_f32(.5f))));
            auto rgb = vcombine_u8(vqmovn_u16(vcombine_u16(a, b)), vqmovn_u16(vcombine_u16(c, c)));
            auto out = vqtbl1q_u8(rgb, indices);
            if (x + i * 4 + 3 < w) {
                vst1q_u8(dst, vbslq_u8(mask, vld1q_u8(dst), out));
            } else if (x + i * 4 < w) {
                uint8_t tail[16];
                vst1q_u8(tail, out);
                for (unsigned j = 0; j < w - x - i * 4; ++j)
                    memcpy(dst + j * 4, tail + j * 4, 3);
            }
        }
    }
}
#else
struct State {
    float value, velocity = 0, acceleration = 0;
    explicit State(float edge) : value(edge) {}
    float step(float x, const Coefficients &k) {
        acceleration = (x - value) * k.gain + acceleration * k.acceleration + velocity * k.velocity;
        velocity += acceleration;
        value += velocity;
        return value;
    }
};
void horizontal(uint8_t *pixels, float *buffer, unsigned w, unsigned h, size_t stride,
                unsigned firstRow, const Coefficients k) {
    thread_local std::vector<float> scratch;
    scratch.resize(size_t(w) * 3);
    float *rowScratch = scratch.data();
    for (unsigned y = firstRow; y < std::min(h, firstRow + rowCount); ++y) {
        uint8_t *row = pixels + y * stride;
        for (unsigned c = 0; c < 3; ++c) {
            State state(row[c]);
            for (unsigned x = 0; x < w; ++x)
                rowScratch[x * 3 + c] = state.step(row[x * 4 + c], k);
            state = State(state.value);
            for (unsigned x = w; x-- > 0;) {
                size_t index = (size_t(x / stripWidth) * ((h + 3) & ~3u) + y) * stripFloats +
                               (x % stripWidth) * 3 + c;
                buffer[index] = state.step(rowScratch[x * 3 + c], k);
            }
        }
    }
}
void vertical(uint8_t *pixels, float *buffer, unsigned w, unsigned h, size_t stride,
              unsigned firstCol, const Coefficients k) {
    float *base = buffer + size_t(firstCol / stripWidth) * ((h + 3) & ~3u) * stripFloats;
    for (unsigned x = firstCol; x < std::min(w, firstCol + stripWidth); ++x) {
        for (unsigned c = 0; c < 3; ++c) {
            float *column = base + (x - firstCol) * 3 + c;
            State state(column[size_t(h - 1) * stripFloats]);
            for (unsigned y = h; y-- > 0;)
                column[size_t(y) * stripFloats] = state.step(column[size_t(y) * stripFloats], k);
            state = State(state.value);
            for (unsigned y = 0; y < h; ++y) {
                float value = state.step(column[size_t(y) * stripFloats], k);
                pixels[y * stride + x * 4 + c] =
                    static_cast<uint8_t>(std::max(0.f, std::min(255.f, value + .5f)));
            }
        }
    }
}

#endif

Executor &sharedExecutor() {
    static Executor executor;
    return executor;
}

template <class F> void parallelFor(unsigned count, bool parallel, F &&f) {
    if (!parallel) {
        for (unsigned i = 0; i < count; ++i)
            f(i);
    } else {
        sharedExecutor().run(count, true, std::forward<F>(f));
    }
}

} // namespace

bool blur(uint8_t *pixels, unsigned width, unsigned height, size_t stride, float sigma) try {
    if (!pixels || !width || !height || stride < size_t(width) * 4 || !std::isfinite(sigma) ||
        sigma < 0)
        return false;
    if (width > std::numeric_limits<unsigned>::max() / 4 ||
        height > std::numeric_limits<unsigned>::max() - 3 ||
        stride > std::numeric_limits<size_t>::max() / height ||
        size_t(width + stripWidth - 1) / stripWidth >
            std::numeric_limits<size_t>::max() / ((height + 3) & ~3u) / stripFloats / sizeof(float))
        return false;
    if (sigma == 0)
        return true;
    Coefficients k(sigma);
    thread_local std::vector<float> buffer;
    buffer.resize(
        size_t((width + stripWidth - 1) / stripWidth) * ((height + 3) & ~3u) * stripFloats + 16);
    // Align both strip bases and four-row task boundaries to cache lines.
    // Adjacent workers can then write without invalidating each other's lines.
    float *data = reinterpret_cast<float *>((reinterpret_cast<uintptr_t>(buffer.data()) + 63) &
                                            ~uintptr_t(63));
    bool parallel = size_t(width) * height >= 16384;
    parallelFor((height + rowCount - 1) / rowCount, parallel, [&](unsigned job) {
        horizontal(pixels, data, width, height, stride, job * rowCount, k);
    });
    parallelFor((width + stripWidth - 1) / stripWidth, parallel, [&](unsigned job) {
        vertical(pixels, data, width, height, stride, job * stripWidth, k);
    });
    return true;
} catch (const std::bad_alloc &) {
    return false;
} catch (const std::length_error &) {
    return false;
}
} // namespace iirblur
