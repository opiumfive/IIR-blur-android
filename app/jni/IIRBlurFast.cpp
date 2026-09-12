#include "IIRBlur.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <cstdio>
#include <sched.h>
#include <unistd.h>
#endif
#ifdef IIRBLUR_PROFILE
#include <cstdlib>
#endif

#if defined(__aarch64__) && !defined(IIRBLUR_FORCE_SCALAR)
#include <arm_neon.h>
#define IIRBLUR_NEON 1
#endif

namespace iirblur {
namespace {
// Both passes are memory bound on phones, so the design minimizes DRAM
// traffic. The image is filtered horizontally in blocks of four rows and
// vertically in strips of eight columns.
//
// Fast mode keeps the horizontally filtered image in the bitmap itself, as
// 8-bit RGB with the alpha untouched: four passes over the pixels and no
// intermediate buffer at all. The half-level rounding of the intermediate
// still keeps the result within one level of the double-precision reference.
//
// Precise mode stores the intermediate as int16 fixed point (6 fractional
// bits) in a planar strip layout
//   strip s, row block b, channel c, row r, column j ->
//   ((s * blocks + b) * 3 + c) * 32 + r * 8 + j
// so a horizontal task writes whole 192-byte blocks and a vertical task streams
// through contiguous memory. It also carries the overshoot of very large
// sigmas, which 8 bits would clip.
constexpr unsigned stripWidth = 8;
constexpr unsigned blockRows = 4;
constexpr unsigned taskStrips = 2; // a vertical task covers 16 columns: full cache lines
constexpr unsigned blockValues = 3 * blockRows * stripWidth; // int16 per (strip, block)
constexpr int fixedBits = 6;
constexpr float fixedScale = 64.f;
// Beyond this sigma the recursive filter overshoots by more than 0.02 levels
// at edges, which 8-bit storage would clip.
constexpr float packedSigmaLimit = 128.f;

// Spinning workers wait in WFE: the core is runnable for the scheduler, so
// its cluster keeps its clock, yet it draws little power until an event or the
// timer event stream wakes it. Signals from the dispatcher use SEV.
inline void cpuRelax() {
#if defined(__aarch64__)
    asm volatile("wfe" ::: "memory");
#elif defined(__i386__) || defined(__x86_64__)
    asm volatile("pause" ::: "memory");
#endif
}
inline void cpuSignal() {
#if defined(__aarch64__)
    asm volatile("sev" ::: "memory");
#endif
}

// How long workers stay awake after a job. Mobile CPUs and memory controllers
// drop their clocks within a few milliseconds of idling and take longer than a
// whole blur to ramp back up, so a blur called once per frame would run at
// idle clocks. The default covers a 60 Hz frame; see setIdleSpin().
std::atomic<unsigned> idleSpinMs{20};

// Android's scheduler places a thread that has just woken from a long sleep by
// its decayed utilization, i.e. on a little core, and migrates it only after a
// few milliseconds: longer than a whole blur. Binding workers to a core class
// keeps them where they were measured. Per-core capacities come from sysfs.
struct Topology {
    unsigned cpus = 0, big = 0;
#if defined(__linux__)
    cpu_set_t bigMask, littleMask;
#endif
    bool usable = false;
    Topology() {
#if defined(__linux__)
        long n = sysconf(_SC_NPROCESSORS_CONF);
        if (n < 2 || n > CPU_SETSIZE)
            return;
        std::vector<long> capacity(size_t(n), 0);
        long maximum = 0;
        for (long i = 0; i < n; ++i) {
            char path[96];
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%ld/cpu_capacity", i);
            FILE *file = fopen(path, "r");
            if (!file)
                return;
            long value = 0;
            bool ok = fscanf(file, "%ld", &value) == 1;
            fclose(file);
            if (!ok)
                return;
            capacity[size_t(i)] = value;
            maximum = std::max(maximum, value);
        }
        CPU_ZERO(&bigMask);
        CPU_ZERO(&littleMask);
        for (long i = 0; i < n; ++i) {
            if (capacity[size_t(i)] * 2 >= maximum) {
                CPU_SET(i, &bigMask);
                ++big;
            } else {
                CPU_SET(i, &littleMask);
            }
        }
        cpus = unsigned(n);
        usable = big > 0 && big < cpus;
#endif
    }
};

thread_local bool littleWorker = false;
#ifdef IIRBLUR_PROFILE
unsigned prefetchRows = getenv("PREFETCH") ? unsigned(atoi(getenv("PREFETCH"))) : 16;
#define PREFETCH_ROWS prefetchRows
#else
#define PREFETCH_ROWS 16
#endif

// Reuses a bounded set of workers. Workers stay awake for a while after each
// job so that the vertical phase and the next animation frame pay neither a
// futex wake-up nor a clock ramp; afterwards they sleep. A job is closed once
// the caller runs out of tasks: only workers that joined it are waited for.
class Executor {
  public:
    Executor() {
        unsigned n = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
        Topology topology;
#ifdef IIRBLUR_PROFILE
        if (getenv("THREADS"))
            n = std::max(1u, unsigned(atoi(getenv("THREADS"))));
        if (getenv("AFFINITY") && atoi(getenv("AFFINITY")) == 0)
            topology.usable = false;
        if (getenv("SPIN_MS"))
            idleSpinMs.store(unsigned(atoi(getenv("SPIN_MS"))));
        if (getenv("RESERVE"))
            reserveTasks = unsigned(atoi(getenv("RESERVE")));
        if (getenv("NO_IDLE_POLICY"))
            policySwitch = false;
#endif
        if (policySwitch)
            policySwitch = probePolicySwitch();
        bigThreads = topology.usable ? std::min(n, topology.big) : n;
        workers.reserve(n - 1);
        for (unsigned i = 1; i < n; ++i) {
            try {
                // The caller counts as one big-core participant.
                bool little = topology.usable && i >= bigThreads;
                workers.emplace_back([this, topology, little, i] {
#if defined(__linux__)
                    if (topology.usable) {
                        const cpu_set_t &mask = little ? topology.littleMask : topology.bigMask;
                        // Cpuset limits (background apps) may reject the mask; the
                        // kernel then keeps placing the thread and nothing else changes.
                        if (sched_setaffinity(0, sizeof mask, &mask) == 0)
                            littleWorker = little;
                    }
#endif
                    worker(i);
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
            generation.fetch_add(1, std::memory_order_release);
        }
        wake.notify_all();
        for (auto &worker : workers)
            worker.join();
    }
    // Slots 0..slots()-1 identify the participating threads; slot 0 is the caller.
    unsigned slots() const { return unsigned(workers.size()) + 1; }
    // f(index, slot) for index in [0, count). Serial when parallel is false.
    template <class F> void run(unsigned count, bool parallel, F &&f) {
        if (!parallel || count < 2 || workers.empty()) {
            for (unsigned i = 0; i < count; ++i)
                f(i, 0);
            return;
        }
        std::lock_guard<std::mutex> dispatch(dispatchMutex);
        std::atomic<unsigned> next{0};
        std::exception_ptr error;
        std::mutex errorMutex;
        // Little cores are several times slower per task. They stop claiming
        // while the big cores still have enough work to outlast a little task,
        // so a slow straggler never delays the completion barrier.
        const unsigned reserve = reserveTasks * bigThreads;
        auto drain = [&](unsigned slot) {
            try {
                for (;;) {
                    if (littleWorker &&
                        count - std::min(count, next.load(std::memory_order_relaxed)) <= reserve)
                        break;
                    unsigned i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= count)
                        break;
                    f(i, slot);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(errorMutex);
                if (!error)
                    error = std::current_exception();
                next.store(count, std::memory_order_relaxed);
            }
        };
        Job job{&drain, [](void *arg, unsigned slot) {
                    (*static_cast<decltype(drain) *>(arg))(slot);
                }};
        // Publish the job, then admit workers. A worker late from the previous
        // job joins through `active` alone, so the descriptor must already be
        // there. Reopen by subtraction: such a worker may add and subtract one
        // concurrently, which a store would lose.
        current = &job;
        active.fetch_sub(closed, std::memory_order_acq_rel);
        {
            std::lock_guard<std::mutex> lock(mutex);
            generation.fetch_add(1, std::memory_order_release);
        }
        cpuSignal();
        wake.notify_all();
        drain(0); // The calling thread contributes work instead of sleeping.
        // Close the job: workers arriving from now on back out without touching
        // it, so a worker that is asleep or starved of CPU time never delays
        // completion. Only workers already inside are waited for.
        active.fetch_add(closed, std::memory_order_acq_rel);
        auto start = std::chrono::steady_clock::now();
        const std::chrono::milliseconds spinLimit{idleSpinMs.load()};
        while (active.load(std::memory_order_acquire) != closed) {
            if (spinLimit.count() == 0 || std::chrono::steady_clock::now() - start > spinLimit) {
                std::unique_lock<std::mutex> lock(mutex);
                done.wait(lock, [&] { return active.load(std::memory_order_acquire) == closed; });
                break;
            }
            cpuRelax();
        }
        current = nullptr;
        if (error)
            std::rethrow_exception(error);
    }

  private:
    struct Job {
        void *arg;
        void (*fn)(void *, unsigned);
    };
    // While spinning, a worker runs as SCHED_IDLE: it only gets a core nobody
    // else wants, so it cannot delay the app's own threads, yet the core stays
    // busy for the frequency governor. Work runs at the normal policy. The
    // switch back needs RLIMIT_NICE headroom; a throwaway thread probes the
    // round trip first, so no worker can ever be left at idle priority.
    static bool probePolicySwitch() {
#if defined(__linux__) && defined(SCHED_IDLE)
        std::atomic<bool> ok{false};
        try {
            std::thread([&] {
                sched_param param{};
                ok = sched_setscheduler(0, SCHED_IDLE, &param) == 0 &&
                     sched_setscheduler(0, SCHED_OTHER, &param) == 0;
            }).join();
        } catch (const std::exception &) {
        }
        return ok;
#else
        return false;
#endif
    }
    bool idlePolicy(bool idle) {
#if defined(__linux__) && defined(SCHED_IDLE)
        if (!policySwitch)
            return false;
        sched_param param{};
        if (sched_setscheduler(0, idle ? SCHED_IDLE : SCHED_OTHER, &param) == 0)
            return true;
        policySwitch = false;
        return false;
#else
        (void)idle;
        return false;
#endif
    }
    // Leaves the current job; the last one out wakes the waiting caller. Used
    // both after working and when backing out of a closed job: a back-out can
    // be the last decrement too.
    void leave() {
        if (active.fetch_sub(1, std::memory_order_acq_rel) == closed + 1) {
            cpuSignal();
            std::lock_guard<std::mutex> lock(mutex);
            done.notify_one();
        }
    }
    void worker(unsigned slot) {
        size_t seen = 0;
        for (;;) {
            size_t g;
            auto start = std::chrono::steady_clock::now();
            const std::chrono::milliseconds spinLimit{idleSpinMs.load()};
            bool lowered = false;
            for (;;) {
                g = generation.load(std::memory_order_acquire);
                if (g != seen)
                    break;
                if (spinLimit.count() == 0 ||
                    std::chrono::steady_clock::now() - start > spinLimit) {
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait(lock, [&] {
                        return generation.load(std::memory_order_acquire) != seen;
                    });
                    g = generation.load(std::memory_order_acquire);
                    break;
                }
                if (!lowered)
                    lowered = idlePolicy(true);
                cpuRelax();
            }
            if (lowered && !idlePolicy(false))
                policySwitch = false; // stuck at SCHED_IDLE: stop lowering, keep working
            seen = g;
            if (stopping)
                return;
            // Join the job unless it has already been closed.
            if (active.fetch_add(1, std::memory_order_acq_rel) >= closed) {
                leave();
                continue;
            }
            Job *job = current;
            job->fn(job->arg, slot);
            leave();
        }
    }
    std::vector<std::thread> workers;
    unsigned bigThreads = 1;
    unsigned reserveTasks = 8;
    std::atomic<bool> policySwitch{true};
    std::mutex mutex, dispatchMutex;
    std::condition_variable wake, done;
    std::atomic<size_t> generation{0};
    static constexpr long closed = 1L << 20;
    std::atomic<long> active{closed}; // workers inside the job, plus closed once it is closed
    Job *current = nullptr;
    bool stopping = false;
};

constexpr long Executor::closed;

Executor &sharedExecutor() {
    static Executor executor;
    return executor;
}

struct Coefficients {
    // Delta realization: acceleration, velocity, value. Avoids cancellation of
    // nearly equal feedback terms and preserves constant colours exactly.
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

struct Plan {
    uint8_t *pixels;
    int16_t *strips; // null in fast mode
    float *scratch;  // one region of scratchFloats per executor slot
    size_t scratchFloats, stride, stripStride;
    unsigned width, height, strips_, blocks, padded;
    Coefficients k;
    bool packed;
    float *scratchFor(unsigned slot) const { return scratch + size_t(slot) * scratchFloats; }
};

#ifdef IIRBLUR_NEON
// All three coefficients in one register: fmla by lane keeps them out of the
// general register budget of the eight-column vertical loop.
inline float32x4_t coefficientVector(const Coefficients &k) {
    float32x4_t v = vdupq_n_f32(0);
    v = vsetq_lane_f32(k.acceleration, v, 0);
    v = vsetq_lane_f32(k.velocity, v, 1);
    v = vsetq_lane_f32(k.gain, v, 2);
    return v;
}
struct State {
    float32x4_t value, velocity, acceleration;
    State() = default;
    explicit State(float32x4_t edge)
        : value(edge), velocity(vdupq_n_f32(0)), acceleration(vdupq_n_f32(0)) {}
    inline float32x4_t step(float32x4_t x, float32x4_t k) {
        auto force = vfmaq_laneq_f32(vmulq_laneq_f32(acceleration, k, 0), velocity, k, 1);
        acceleration = vfmaq_laneq_f32(force, vsubq_f32(x, value), k, 2);
        velocity = vaddq_f32(velocity, acceleration);
        value = vaddq_f32(value, velocity);
        return value;
    }
};

// Byte shuffles that place one channel of four pixels into the second byte of
// each 32-bit lane, i.e. the channel value times 256; a fixed point conversion
// then divides by 256 (fast mode) or by 4 (precise mode: values times 64).
alignas(16) const uint8_t channelTable[3][16] = {
    {255, 0, 255, 255, 255, 4, 255, 255, 255, 8, 255, 255, 255, 12, 255, 255},
    {255, 1, 255, 255, 255, 5, 255, 255, 255, 9, 255, 255, 255, 13, 255, 255},
    {255, 2, 255, 255, 255, 6, 255, 255, 255, 10, 255, 255, 255, 14, 255, 255}};
// Interleave {R0..R7, G0..G7} and {B0..B7, B0..B7} into RGBx pixels.
alignas(16) const uint8_t packTable[2][16] = {
    {0, 8, 16, 255, 1, 9, 17, 255, 2, 10, 18, 255, 3, 11, 19, 255},
    {4, 12, 20, 255, 5, 13, 21, 255, 6, 14, 22, 255, 7, 15, 23, 255}};
// From {c0: r0 r1 r2 r3, c1: .., c2: .., c3: ..} per channel (three table
// registers R, G, B) to the RGBx pixels of row r.
alignas(16) const uint8_t rowTable[blockRows][16] = {
    {0, 16, 32, 255, 4, 20, 36, 255, 8, 24, 40, 255, 12, 28, 44, 255},
    {1, 17, 33, 255, 5, 21, 37, 255, 9, 25, 41, 255, 13, 29, 45, 255},
    {2, 18, 34, 255, 6, 22, 38, 255, 10, 26, 42, 255, 14, 30, 46, 255},
    {3, 19, 35, 255, 7, 23, 39, 255, 11, 27, 43, 255, 15, 31, 47, 255}};

template <int fbits> inline float32x4_t channel(uint8x16_t column, uint8x16_t table) {
    return vcvtq_n_f32_u32(vreinterpretq_u32_u8(vqtbl1q_u8(column, table)), fbits);
}

// Stores four RGB pixels over the destination, keeping its alpha bytes.
inline void storePixels(uint8_t *dst, uint8x16_t rgb, unsigned count) {
    const uint8x16_t alphaMask = vreinterpretq_u8_u32(vdupq_n_u32(0xff000000u));
    if (count >= 4) {
        vst1q_u8(dst, vbslq_u8(alphaMask, vld1q_u8(dst), rgb));
    } else {
        alignas(16) uint8_t tail[16];
        vst1q_u8(tail, rgb);
        for (unsigned j = 0; j < count; ++j)
            memcpy(dst + j * 4, tail + j * 4, 3);
    }
}

// Forward horizontal sweep of four rows: rows in lanes, one recurrence per
// channel. Writes w columns of {R, G, B} vectors, then replicates the last
// column up to the padded width: with zero derivatives the recurrence holds
// it exactly, so the reverse sweep may start at the padding and reach column
// w - 1 in the proper edge state.
template <int fbits>
void horizontalForward(const uint8_t *const rows[blockRows], unsigned w, unsigned padded,
                       float *rowScratch, const float32x4_t k) {
    const uint8x16_t tableR = vld1q_u8(channelTable[0]);
    const uint8x16_t tableG = vld1q_u8(channelTable[1]);
    const uint8x16_t tableB = vld1q_u8(channelTable[2]);
    uint32x4_t edge = vdupq_n_u32(0);
    for (unsigned r = 0; r < blockRows; ++r) {
        uint32_t pixel;
        memcpy(&pixel, rows[r], 4);
        edge = vsetq_lane_u32(pixel, edge, 0);
        edge = vextq_u32(edge, edge, 1); // rotate: lane r ends up holding row r
    }
    State sR(channel<fbits>(vreinterpretq_u8_u32(edge), tableR));
    State sG(channel<fbits>(vreinterpretq_u8_u32(edge), tableG));
    State sB(channel<fbits>(vreinterpretq_u8_u32(edge), tableB));
    alignas(16) uint8_t tail[blockRows][16] = {};
    for (unsigned x = 0; x < w; x += 4) {
        const uint8_t *p[blockRows];
        if (x + 4 <= w) {
            for (unsigned r = 0; r < blockRows; ++r)
                p[r] = rows[r] + x * 4;
        } else {
            for (unsigned r = 0; r < blockRows; ++r) {
                memcpy(tail[r], rows[r] + x * 4, (w - x) * 4);
                p[r] = tail[r];
            }
        }
        // In-order little cores stall on these four strided streams without
        // explicit prefetch; big cores' prefetchers already run ahead.
        __builtin_prefetch(p[0] + 128);
        __builtin_prefetch(p[1] + 128);
        __builtin_prefetch(p[2] + 128);
        __builtin_prefetch(p[3] + 128);
        uint32x4_t a0 = vreinterpretq_u32_u8(vld1q_u8(p[0]));
        uint32x4_t a1 = vreinterpretq_u32_u8(vld1q_u8(p[1]));
        uint32x4_t a2 = vreinterpretq_u32_u8(vld1q_u8(p[2]));
        uint32x4_t a3 = vreinterpretq_u32_u8(vld1q_u8(p[3]));
        uint32x4_t t0 = vtrn1q_u32(a0, a1), t1 = vtrn2q_u32(a0, a1);
        uint32x4_t t2 = vtrn1q_u32(a2, a3), t3 = vtrn2q_u32(a2, a3);
        uint8x16_t c[4] = {
            vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2))),
            vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3))),
            vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2))),
            vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)))};
        float *out = rowScratch + size_t(x) * 12;
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            vst1q_f32(out + j * 12, sR.step(channel<fbits>(c[j], tableR), k));
            vst1q_f32(out + j * 12 + 4, sG.step(channel<fbits>(c[j], tableG), k));
            vst1q_f32(out + j * 12 + 8, sB.step(channel<fbits>(c[j], tableB), k));
        }
    }
    for (unsigned p = w; p < padded; ++p)
        memcpy(rowScratch + size_t(p) * 12, rowScratch + size_t(w - 1) * 12, 12 * sizeof(float));
}

// Two filtered columns of four rows as one vector of fixed point values,
// {c0: r0 r1 r2 r3, c1: r0 r1 r2 r3}.
inline int16x8_t narrowPair(float32x4_t c0, float32x4_t c1) {
    return vqmovn_high_s32(vqmovn_s32(vcvtaq_s32_f32(c0)), vcvtaq_s32_f32(c1));
}
// Eight columns (four such pairs) transposed into the four rows of a block.
inline void storeBlock(int16_t *dst, int16x8_t p01, int16x8_t p23, int16x8_t p45,
                       int16x8_t p67) {
    int16x8_t u0 = vuzp1q_s16(p01, p23), u1 = vuzp2q_s16(p01, p23);
    int16x8_t u2 = vuzp1q_s16(p45, p67), u3 = vuzp2q_s16(p45, p67);
    vst1q_s16(dst, vuzp1q_s16(u0, u2));
    vst1q_s16(dst + 8, vuzp1q_s16(u1, u3));
    vst1q_s16(dst + 16, vuzp2q_s16(u0, u2));
    vst1q_s16(dst + 24, vuzp2q_s16(u1, u3));
}
// Four columns of four rows as bytes, {c0: r0 r1 r2 r3, c1: .., c2: .., c3: ..}.
inline uint8x16_t narrowColumns(float32x4_t c0, float32x4_t c1, float32x4_t c2,
                                float32x4_t c3) {
    uint16x8_t p01 = vqmovun_high_s32(vqmovun_s32(vcvtaq_s32_f32(c0)), vcvtaq_s32_f32(c1));
    uint16x8_t p23 = vqmovun_high_s32(vqmovun_s32(vcvtaq_s32_f32(c2)), vcvtaq_s32_f32(c3));
    return vqmovn_high_u16(vqmovn_u16(p01), p23);
}

void horizontalTask(const Plan &p, unsigned block, float *rowScratch) {
    const unsigned w = p.width, y0 = block * blockRows;
    const uint8_t *rows[blockRows];
    for (unsigned r = 0; r < blockRows; ++r)
        rows[r] = p.pixels + size_t(std::min(y0 + r, p.height - 1)) * p.stride;
    const float32x4_t k = coefficientVector(p.k);
    if (p.packed)
        horizontalForward<8>(rows, w, p.padded, rowScratch, k);
    else
        horizontalForward<2>(rows, w, p.padded, rowScratch, k);
    // The reverse sweep restarts from the last real column, not from the tail
    // group's zero padding, one strip (eight columns) at a time. Results are
    // narrowed as soon as they exist to keep the register budget.
    const float *edge = rowScratch + size_t(w - 1) * 12;
    State sR(vld1q_f32(edge)), sG(vld1q_f32(edge + 4)), sB(vld1q_f32(edge + 8));
    const unsigned validRows = std::min(blockRows, p.height - y0);
    for (unsigned s = p.strips_; s-- > 0;) {
        const float *t = rowScratch + size_t(s) * stripWidth * 12;
        const unsigned x0 = s * stripWidth;
        if (s >= 2) // the strip after next, for in-order cores
            for (unsigned line = 0; line < stripWidth * 12; line += 16)
                __builtin_prefetch(t - 2 * stripWidth * 12 + line);
        if (!p.packed) {
            int16x8_t R[4], G[4], B[4];
#pragma unroll
            for (unsigned j = 4; j-- > 0;) {
                float32x4_t r1 = sR.step(vld1q_f32(t + j * 24 + 12), k);
                float32x4_t g1 = sG.step(vld1q_f32(t + j * 24 + 16), k);
                float32x4_t b1 = sB.step(vld1q_f32(t + j * 24 + 20), k);
                R[j] = narrowPair(sR.step(vld1q_f32(t + j * 24), k), r1);
                G[j] = narrowPair(sG.step(vld1q_f32(t + j * 24 + 4), k), g1);
                B[j] = narrowPair(sB.step(vld1q_f32(t + j * 24 + 8), k), b1);
            }
            int16_t *dst = p.strips + size_t(s) * p.stripStride + size_t(block) * blockValues;
            storeBlock(dst, R[0], R[1], R[2], R[3]);
            storeBlock(dst + blockRows * stripWidth, G[0], G[1], G[2], G[3]);
            storeBlock(dst + 2 * blockRows * stripWidth, B[0], B[1], B[2], B[3]);
            continue;
        }
#pragma unroll
        for (unsigned half = 2; half-- > 0;) {
            const float *u = t + half * 48;
            float32x4_t r[4], g[4], b[4];
#pragma unroll
            for (unsigned j = 4; j-- > 0;) {
                r[j] = sR.step(vld1q_f32(u + j * 12), k);
                g[j] = sG.step(vld1q_f32(u + j * 12 + 4), k);
                b[j] = sB.step(vld1q_f32(u + j * 12 + 8), k);
            }
            if (x0 + half * 4 >= w)
                continue;
            uint8x16x3_t table;
            table.val[0] = narrowColumns(r[0], r[1], r[2], r[3]);
            table.val[1] = narrowColumns(g[0], g[1], g[2], g[3]);
            table.val[2] = narrowColumns(b[0], b[1], b[2], b[3]);
            const unsigned count = std::min(4u, w - x0 - half * 4);
            for (unsigned row = 0; row < validRows; ++row) {
                uint8_t *dst = p.pixels + size_t(y0 + row) * p.stride + size_t(x0 + half * 4) * 4;
                storePixels(dst, vqtbl3q_u8(table, vld1q_u8(rowTable[row])), count);
            }
        }
    }
}

// Reads the intermediate for one strip row: six vectors {R0-3 R4-7 G.. B..}.
struct StripSource {
    const int16_t *strip;
    inline void operator()(unsigned y, float32x4_t *out) const {
        const int16_t *q =
            strip + size_t(y / blockRows) * blockValues + (y % blockRows) * stripWidth;
        int16x8_t r = vld1q_s16(q), g = vld1q_s16(q + blockRows * stripWidth),
                  b = vld1q_s16(q + 2 * blockRows * stripWidth);
        out[0] = vcvtq_n_f32_s32(vmovl_s16(vget_low_s16(r)), fixedBits);
        out[1] = vcvtq_n_f32_s32(vmovl_high_s16(r), fixedBits);
        out[2] = vcvtq_n_f32_s32(vmovl_s16(vget_low_s16(g)), fixedBits);
        out[3] = vcvtq_n_f32_s32(vmovl_high_s16(g), fixedBits);
        out[4] = vcvtq_n_f32_s32(vmovl_s16(vget_low_s16(b)), fixedBits);
        out[5] = vcvtq_n_f32_s32(vmovl_high_s16(b), fixedBits);
    }
};
struct PackedSource {
    const uint8_t *column; // pixels + x0 * 4
    size_t stride;
    unsigned valid; // columns present in this strip
    uint8x16_t tableR, tableG, tableB;
    inline void operator()(unsigned y, float32x4_t *out) const {
        const uint8_t *q = column + size_t(y) * stride;
        if (y >= PREFETCH_ROWS)
            __builtin_prefetch(q - PREFETCH_ROWS * stride);
        uint8x16_t a, b;
        if (valid == stripWidth) {
            a = vld1q_u8(q);
            b = vld1q_u8(q + 16);
        } else {
            alignas(16) uint8_t tail[32] = {};
            memcpy(tail, q, valid * 4);
            a = vld1q_u8(tail);
            b = vld1q_u8(tail + 16);
        }
        out[0] = channel<8>(a, tableR);
        out[1] = channel<8>(b, tableR);
        out[2] = channel<8>(a, tableG);
        out[3] = channel<8>(b, tableG);
        out[4] = channel<8>(a, tableB);
        out[5] = channel<8>(b, tableB);
    }
};

template <class Source>
void verticalStrip(const Source &source, uint8_t *column, size_t stride, unsigned h,
                   unsigned valid, float *colScratch, const float32x4_t k) {
    constexpr unsigned lanes = 3 * stripWidth / 4;
    float32x4_t in[lanes];
    State st[lanes];
    source(h - 1, in);
#pragma unroll
    for (unsigned c = 0; c < lanes; ++c)
        st[c] = State(in[c]);
    for (unsigned y = h; y-- > 0;) {
        source(y, in);
        float *t = colScratch + size_t(y) * lanes * 4;
#pragma unroll
        for (unsigned c = 0; c < lanes; ++c)
            vst1q_f32(t + c * 4, st[c].step(in[c], k));
    }
#pragma unroll
    for (unsigned c = 0; c < lanes; ++c)
        st[c] = State(st[c].value);
    const uint8x16_t pack0 = vld1q_u8(packTable[0]), pack1 = vld1q_u8(packTable[1]);
    uint8_t *dst = column;
    for (unsigned y = 0; y < h; ++y, dst += stride) {
        if (y + PREFETCH_ROWS < h)
            __builtin_prefetch(dst + PREFETCH_ROWS * stride, 1);
        const float *t = colScratch + size_t(y) * lanes * 4;
        int32x4_t v[lanes];
#pragma unroll
        for (unsigned c = 0; c < lanes; ++c)
            v[c] = vcvtaq_s32_f32(st[c].step(vld1q_f32(t + c * 4), k));
        uint16x8_t r = vqmovun_high_s32(vqmovun_s32(v[0]), v[1]);
        uint16x8_t g = vqmovun_high_s32(vqmovun_s32(v[2]), v[3]);
        uint16x8_t b = vqmovun_high_s32(vqmovun_s32(v[4]), v[5]);
        uint8x16x2_t table;
        table.val[0] = vqmovn_high_u16(vqmovn_u16(r), g);
        uint8x8_t b8 = vqmovn_u16(b);
        table.val[1] = vcombine_u8(b8, b8);
        storePixels(dst, vqtbl2q_u8(table, pack0), valid);
        if (valid > 4)
            storePixels(dst + 16, vqtbl2q_u8(table, pack1), valid - 4);
    }
}

void verticalTask(const Plan &p, unsigned task, float *colScratch) {
    const float32x4_t k = coefficientVector(p.k);
    for (unsigned s = task * taskStrips; s < std::min(p.strips_, (task + 1) * taskStrips); ++s) {
        const unsigned x0 = s * stripWidth, valid = std::min(stripWidth, p.width - x0);
        uint8_t *column = p.pixels + size_t(x0) * 4;
        if (p.packed) {
            PackedSource source{column,
                                p.stride,
                                valid,
                                vld1q_u8(channelTable[0]),
                                vld1q_u8(channelTable[1]),
                                vld1q_u8(channelTable[2])};
            verticalStrip(source, column, p.stride, p.height, valid, colScratch, k);
        } else {
            StripSource source{p.strips + size_t(s) * p.stripStride};
            verticalStrip(source, column, p.stride, p.height, valid, colScratch, k);
        }
    }
}
#else
struct State {
    float value, velocity = 0, acceleration = 0;
    State() = default;
    explicit State(float edge) : value(edge) {}
    float step(float x, const Coefficients &k) {
        acceleration = (x - value) * k.gain + acceleration * k.acceleration + velocity * k.velocity;
        velocity += acceleration;
        value += velocity;
        return value;
    }
};
inline uint8_t toByte(float v) {
    return uint8_t(std::max(0.f, std::min(255.f, std::nearbyint(v))));
}
inline int16_t toFixed(float v) {
    return int16_t(std::max(-32768.f, std::min(32767.f, std::nearbyint(v * fixedScale))));
}
void horizontalTask(const Plan &p, unsigned block, float *rowScratch) {
    const unsigned w = p.width, y0 = block * blockRows;
    for (unsigned r = 0; r < blockRows; ++r) {
        uint8_t *row = p.pixels + size_t(std::min(y0 + r, p.height - 1)) * p.stride;
        for (unsigned c = 0; c < 3; ++c) {
            State state(row[c]);
            for (unsigned x = 0; x < w; ++x)
                rowScratch[x] = state.step(row[x * 4 + c], p.k);
            state = State(state.value);
            for (unsigned x = w; x-- > 0;) {
                float v = state.step(rowScratch[x], p.k);
                if (p.packed) {
                    if (y0 + r < p.height)
                        row[x * 4 + c] = toByte(v);
                } else {
                    p.strips[size_t(x / stripWidth) * p.stripStride + size_t(block) * blockValues +
                             c * blockRows * stripWidth + r * stripWidth + x % stripWidth] =
                        toFixed(v);
                }
            }
        }
    }
}
void verticalTask(const Plan &p, unsigned task, float *colScratch) {
    const unsigned h = p.height;
    for (unsigned s = task * taskStrips; s < std::min(p.strips_, (task + 1) * taskStrips); ++s) {
        const unsigned x0 = s * stripWidth, valid = std::min(stripWidth, p.width - x0);
        const int16_t *strip = p.strips ? p.strips + size_t(s) * p.stripStride : nullptr;
        for (unsigned j = 0; j < valid; ++j)
            for (unsigned c = 0; c < 3; ++c) {
                uint8_t *column = p.pixels + size_t(x0 + j) * 4 + c;
                auto at = [&](unsigned y) {
                    if (p.packed)
                        return float(column[size_t(y) * p.stride]);
                    return strip[size_t(y / blockRows) * blockValues + c * blockRows * stripWidth +
                                 (y % blockRows) * stripWidth + j] /
                           fixedScale;
                };
                State state(at(h - 1));
                for (unsigned y = h; y-- > 0;)
                    colScratch[y] = state.step(at(y), p.k);
                state = State(state.value);
                for (unsigned y = 0; y < h; ++y)
                    column[size_t(y) * p.stride] = toByte(state.step(colScratch[y], p.k));
            }
    }
}
#endif


// ---------------------------------------------------------------------------
// Draft mode: blur a box-downsampled copy and interpolate it back. For a
// Gaussian of width sigma the reconstruction error of a factor f is about
// 7.7 f^2 / sigma^2 levels, so f <= sigma / 12 keeps it under a tenth of a
// level on smooth content. The box filter and the bilinear tent add f^2 / 4 to
// the variance, which the small blur's sigma compensates.
// ---------------------------------------------------------------------------
struct DraftGeometry {
    unsigned factor, shift; // factor = 1 << shift
    unsigned smallWidth, smallHeight;
    size_t smallStride; // bytes
};

inline unsigned draftFactor(float sigma, unsigned width, unsigned height) {
    unsigned f = 1;
    while (f < 8 && sigma >= 12.f * f && width >= 16 * f && height >= 16 * f)
        f *= 2;
    return f;
}

// Vertical interpolation for output row y: source small rows r0, r1 and the
// weight of r1 in units of 1 / (2 f). Pixel centres: small pixel j covers
// output rows f j .. f j + f - 1, so its centre is at f j + (f - 1) / 2.
inline void draftRows(unsigned y, const DraftGeometry &d, unsigned &r0, unsigned &r1,
                      unsigned &weight1) {
    const long twoF = 2 * long(d.factor);
    const long num = 2 * long(y) - long(d.factor) + 1; // 2f * (y - (f-1)/2) / f
    if (num < 0) {
        r0 = r1 = 0;
        weight1 = 0;
        return;
    }
    const long j0 = num / twoF;
    if (j0 + 1 >= long(d.smallHeight)) {
        r0 = r1 = d.smallHeight - 1;
        weight1 = 0;
        return;
    }
    r0 = unsigned(j0);
    r1 = r0 + 1;
    weight1 = unsigned(num % twoF);
}
// Horizontal interpolation for output phase p (x = f i + p): source offset
// (-1 or 0 relative to i) and the weight of the right neighbour in 1 / (2 f).
inline void draftPhase(unsigned phase, unsigned f, int &offset, unsigned &weight1) {
    const long num = 2 * long(phase) - long(f) + 1;
    offset = num < 0 ? -1 : 0;
    weight1 = unsigned(num < 0 ? num + 2 * long(f) : num);
}

#ifdef IIRBLUR_NEON
// Averages factor x factor blocks of source rows y0 .. y0 + factor - 1 into one
// small row; returns whether every alpha byte in those rows was 255.
bool downsampleRow(const uint8_t *pixels, size_t stride, unsigned w, unsigned h,
                   const DraftGeometry &d, unsigned smallRow, uint16_t *acc, uint8_t *out) {
    const unsigned f = d.factor, y0 = smallRow * f, rows = std::min(f, h - y0);
    const unsigned bytes = w * 4;
    uint8x16_t alphaAll = vdupq_n_u8(0xff);
    for (unsigned r = 0; r < rows; ++r) {
        const uint8_t *row = pixels + size_t(y0 + r) * stride;
        unsigned x = 0;
        for (; x + 16 <= bytes; x += 16) {
            uint8x16_t v = vld1q_u8(row + x);
            alphaAll = vandq_u8(alphaAll, v);
            if (r == 0) {
                vst1q_u16(acc + x, vmovl_u8(vget_low_u8(v)));
                vst1q_u16(acc + x + 8, vmovl_high_u8(v));
            } else {
                vst1q_u16(acc + x, vaddw_u8(vld1q_u16(acc + x), vget_low_u8(v)));
                vst1q_u16(acc + x + 8, vaddw_high_u8(vld1q_u16(acc + x + 8), v));
            }
        }
        for (; x < bytes; ++x) {
            acc[x] = r == 0 ? row[x] : uint16_t(acc[x] + row[x]);
            if (x % 4 == 3 && row[x] != 0xff)
                alphaAll = vdupq_n_u8(0);
        }
    }
    // Horizontal groups of f pixels. Full blocks use a rounding shift; the
    // partial block at the right edge and partial rows divide exactly.
    const unsigned full = w / f;
    unsigned i = 0;
    if (rows == f) {
        for (; i < full; ++i) {
            const uint16_t *q = acc + size_t(i) * f * 4;
            uint16x8_t sum;
            if (f == 2) {
                sum = vld1q_u16(q);
            } else if (f == 4) {
                sum = vaddq_u16(vld1q_u16(q), vld1q_u16(q + 8));
            } else {
                sum = vaddq_u16(vaddq_u16(vld1q_u16(q), vld1q_u16(q + 8)),
                                vaddq_u16(vld1q_u16(q + 16), vld1q_u16(q + 24)));
            }
            uint16x4_t pixel = vadd_u16(vget_low_u16(sum), vget_high_u16(sum));
            uint16x8_t both = vcombine_u16(pixel, pixel);
            uint8x8_t narrow = f == 2   ? vrshrn_n_u16(both, 2)
                               : f == 4 ? vrshrn_n_u16(both, 4)
                                        : vrshrn_n_u16(both, 6);
            uint32_t packed = vget_lane_u32(vreinterpret_u32_u8(narrow), 0);
            memcpy(out + size_t(i) * 4, &packed, 4);
        }
    }
    for (; i < d.smallWidth; ++i) {
        const unsigned cols = std::min(f, w - i * f), count = cols * rows;
        for (unsigned c = 0; c < 4; ++c) {
            unsigned sum = 0;
            for (unsigned j = 0; j < cols; ++j)
                sum += acc[(size_t(i) * f + j) * 4 + c];
            out[size_t(i) * 4 + c] = uint8_t((sum + count / 2) / count);
        }
    }
    const uint8x16_t alphaMask = vreinterpretq_u8_u32(vdupq_n_u32(0xff000000u));
    return vminvq_u8(vorrq_u8(alphaAll, vmvnq_u8(alphaMask))) == 0xff;
}

// Writes output rows y0 .. y0 + count - 1 from the blurred small image. temp
// holds one vertically blended small row (scaled by 2 f) with one replicated
// pixel on each side; it needs (smallWidth + 2) * 4 entries.
void upsampleRows(uint8_t *pixels, size_t stride, unsigned w, unsigned h, const DraftGeometry &d,
                  const uint8_t *small, unsigned y0, unsigned count, bool opaque, uint16_t *temp) {
    const unsigned f = d.factor, twoF = 2 * f, ws = d.smallWidth;
    const uint8x16_t alphaMask = vreinterpretq_u8_u32(vdupq_n_u32(0xff000000u));
    // Per phase pair: source offsets and weights of the two output pixels.
    int offsets[8];
    uint16x8_t weight0[4], weight1[4];
    for (unsigned phase = 0; phase < f; phase += 2) {
        unsigned wa, wb;
        draftPhase(phase, f, offsets[phase], wa);
        draftPhase(phase + 1, f, offsets[phase + 1], wb);
        weight1[phase / 2] = vcombine_u16(vdup_n_u16(uint16_t(wa)), vdup_n_u16(uint16_t(wb)));
        weight0[phase / 2] = vsubq_u16(vdupq_n_u16(uint16_t(twoF)), weight1[phase / 2]);
    }
    uint16_t *t = temp + 4; // t[-4..-1] and t[ws*4 ..] replicate the edge pixels
    for (unsigned y = y0; y < y0 + count; ++y) {
        unsigned r0, r1, wy1;
        draftRows(y, d, r0, r1, wy1);
        const unsigned wy0 = twoF - wy1;
        const uint8_t *s0 = small + size_t(r0) * d.smallStride;
        const uint8_t *s1 = small + size_t(r1) * d.smallStride;
        unsigned i = 0;
        for (; i + 4 <= ws; i += 4) {
            uint8x16_t a = vld1q_u8(s0 + size_t(i) * 4), b = vld1q_u8(s1 + size_t(i) * 4);
            uint16x8_t lo = vmlaq_n_u16(vmulq_n_u16(vmovl_u8(vget_low_u8(a)), uint16_t(wy0)),
                                        vmovl_u8(vget_low_u8(b)), uint16_t(wy1));
            uint16x8_t hi = vmlaq_n_u16(vmulq_n_u16(vmovl_high_u8(a), uint16_t(wy0)),
                                        vmovl_high_u8(b), uint16_t(wy1));
            vst1q_u16(t + size_t(i) * 4, lo);
            vst1q_u16(t + size_t(i) * 4 + 8, hi);
        }
        for (; i < ws; ++i)
            for (unsigned c = 0; c < 4; ++c)
                t[size_t(i) * 4 + c] =
                    uint16_t(s0[size_t(i) * 4 + c] * wy0 + s1[size_t(i) * 4 + c] * wy1);
        memcpy(t - 4, t, 8);
        memcpy(t + size_t(ws) * 4, t + size_t(ws - 1) * 4, 8);
        uint8_t *dst = pixels + size_t(y) * stride;
        for (unsigned i2 = 0; i2 < ws; ++i2) {
            const uint16_t *base = t + size_t(i2) * 4;
            alignas(16) uint8_t block[32];
            for (unsigned phase = 0; phase < f; phase += 2) {
                const uint16_t *pa = base + offsets[phase] * 4, *pb = base + offsets[phase + 1] * 4;
                uint16x8_t src0 = vcombine_u16(vld1_u16(pa), vld1_u16(pb));
                uint16x8_t src1 = vcombine_u16(vld1_u16(pa + 4), vld1_u16(pb + 4));
                uint16x8_t v = vmlaq_u16(vmulq_u16(src0, weight0[phase / 2]), src1,
                                         weight1[phase / 2]);
                uint8x8_t narrow = f == 2   ? vrshrn_n_u16(v, 4)
                                   : f == 4 ? vrshrn_n_u16(v, 6)
                                            : vrshrn_n_u16(v, 8);
                vst1_u8(block + phase * 4, narrow);
            }
            uint8_t *q = dst + size_t(i2) * f * 4;
            const unsigned valid = std::min(f, w - i2 * f);
            if (valid == f && opaque) {
                if (f == 2)
                    vst1_u8(q, vld1_u8(block));
                else
                    for (unsigned j = 0; j < f; j += 4)
                        vst1q_u8(q + j * 4, vld1q_u8(block + j * 4));
            } else if (valid == f) {
                if (f == 2)
                    vst1_u8(q, vbsl_u8(vget_low_u8(alphaMask), vld1_u8(q), vld1_u8(block)));
                else
                    for (unsigned j = 0; j < f; j += 4)
                        vst1q_u8(q + j * 4,
                                 vbslq_u8(alphaMask, vld1q_u8(q + j * 4), vld1q_u8(block + j * 4)));
            } else {
                for (unsigned j = 0; j < valid; ++j)
                    memcpy(q + j * 4, block + j * 4, opaque ? 4 : 3);
            }
        }
    }
}
#else
bool downsampleRow(const uint8_t *pixels, size_t stride, unsigned w, unsigned h,
                   const DraftGeometry &d, unsigned smallRow, uint16_t *, uint8_t *out) {
    const unsigned f = d.factor, y0 = smallRow * f, rows = std::min(f, h - y0);
    bool opaque = true;
    for (unsigned i = 0; i < d.smallWidth; ++i) {
        const unsigned cols = std::min(f, w - i * f), count = cols * rows;
        unsigned sum[4] = {0, 0, 0, 0};
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned j = 0; j < cols; ++j) {
                const uint8_t *px = pixels + size_t(y0 + r) * stride + (size_t(i) * f + j) * 4;
                for (unsigned c = 0; c < 4; ++c)
                    sum[c] += px[c];
                opaque &= px[3] == 0xff;
            }
        for (unsigned c = 0; c < 4; ++c)
            out[size_t(i) * 4 + c] = uint8_t((sum[c] + count / 2) / count);
    }
    return opaque;
}
void upsampleRows(uint8_t *pixels, size_t stride, unsigned w, unsigned h, const DraftGeometry &d,
                  const uint8_t *small, unsigned y0, unsigned count, bool opaque, uint16_t *temp) {
    const unsigned f = d.factor, twoF = 2 * f, ws = d.smallWidth;
    uint16_t *t = temp + 4;
    for (unsigned y = y0; y < y0 + count; ++y) {
        unsigned r0, r1, wy1;
        draftRows(y, d, r0, r1, wy1);
        const uint8_t *s0 = small + size_t(r0) * d.smallStride;
        const uint8_t *s1 = small + size_t(r1) * d.smallStride;
        for (unsigned i = 0; i < ws; ++i)
            for (unsigned c = 0; c < 4; ++c)
                t[size_t(i) * 4 + c] = uint16_t(s0[size_t(i) * 4 + c] * (twoF - wy1) +
                                                s1[size_t(i) * 4 + c] * wy1);
        memcpy(t - 4, t, 8);
        memcpy(t + size_t(ws) * 4, t + size_t(ws - 1) * 4, 8);
        uint8_t *dst = pixels + size_t(y) * stride;
        for (unsigned x = 0; x < w; ++x) {
            int offset;
            unsigned wx1;
            draftPhase(x % f, f, offset, wx1);
            const uint16_t *a = t + (long(x / f) + offset) * 4, *b = a + 4;
            for (unsigned c = 0; c < (opaque ? 4u : 3u); ++c)
                dst[size_t(x) * 4 + c] =
                    uint8_t((a[c] * (twoF - wx1) + b[c] * wx1 + twoF * f) / (twoF * twoF));
        }
    }
}
#endif

struct CallerStorage {
    std::vector<int16_t> strips;
    std::vector<float> scratch;
    std::vector<uint8_t> small;   // draft mode: the downsampled image
    std::vector<uint16_t> rows;   // draft mode: one accumulator row per slot
};
CallerStorage &callerStorage() {
    thread_local CallerStorage storage;
    return storage;
}
template <class T> T *aligned64(std::vector<T> &v, size_t count) {
    if (v.size() < count + 64 / sizeof(T))
        v.resize(count + 64 / sizeof(T));
    return reinterpret_cast<T *>((reinterpret_cast<uintptr_t>(v.data()) + 63) & ~uintptr_t(63));
}

} // namespace

#ifdef IIRBLUR_PROFILE
double profile[3];
#endif

void setIdleSpin(unsigned milliseconds) { idleSpinMs.store(std::min(milliseconds, 1000u)); }

namespace {
bool draftBlur(uint8_t *pixels, unsigned width, unsigned height, size_t stride, float sigma,
               DraftGeometry d) {
    d.smallWidth = (width + d.factor - 1) / d.factor;
    d.smallHeight = (height + d.factor - 1) / d.factor;
    d.smallStride = size_t(d.smallWidth) * 4;
    const bool parallel = size_t(width) * height >= 16384;
    Executor &executor = sharedExecutor();
    const unsigned slots = parallel ? executor.slots() : 1;
    CallerStorage &storage = callerStorage();
    const size_t rowValues = std::max(size_t(width) * 4, (size_t(d.smallWidth) + 2) * 4) + 8;
    uint8_t *small = aligned64(storage.small, d.smallStride * d.smallHeight);
    uint16_t *rows = aligned64(storage.rows, rowValues * slots);
    bool opaqueSlot[64];
    std::fill(opaqueSlot, opaqueSlot + slots, true);
    executor.run(d.smallHeight, parallel, [&](unsigned job, unsigned slot) {
        if (!downsampleRow(pixels, stride, width, height, d, job, rows + rowValues * slot,
                           small + d.smallStride * job))
            opaqueSlot[slot] = false;
    });
    bool opaque = true;
    for (unsigned i = 0; i < slots; ++i)
        opaque &= opaqueSlot[i];
    const float f = float(d.factor);
    const float sigmaSmall = std::sqrt(std::max(sigma * sigma - f * f / 4, .25f)) / f;
    // The small blur reuses this thread's storage, but not the two vectors above.
    if (!blur(small, d.smallWidth, d.smallHeight, d.smallStride, sigmaSmall, Quality::fast))
        return false;
    executor.run(d.smallHeight, parallel, [&](unsigned job, unsigned slot) {
        const unsigned y0 = job * d.factor;
        upsampleRows(pixels, stride, width, height, d, small, y0, std::min(d.factor, height - y0),
                     opaque, rows + rowValues * slot);
    });
    return true;
}
} // namespace

bool blur(uint8_t *pixels, unsigned width, unsigned height, size_t stride, float sigma,
          Quality quality) try {
    if (!pixels || !width || !height || stride < size_t(width) * 4 || !std::isfinite(sigma) ||
        sigma < 0)
        return false;
    if (quality == Quality::draft) {
        DraftGeometry d{draftFactor(sigma, width, height), 0, 0, 0, 0};
        while ((1u << d.shift) < d.factor)
            ++d.shift;
        if (d.factor > 1 && sigma > 0)
            return draftBlur(pixels, width, height, stride, sigma, d);
        quality = Quality::fast;
    }
    constexpr size_t maxSize = std::numeric_limits<size_t>::max();
    if (width > std::numeric_limits<unsigned>::max() - stripWidth ||
        height > std::numeric_limits<unsigned>::max() - blockRows ||
        stride > maxSize / height)
        return false;
    Plan p{pixels, nullptr, nullptr, 0, stride, 0, width, height, 0, 0, 0, Coefficients(sigma),
           quality == Quality::fast || (quality == Quality::automatic && sigma <= packedSigmaLimit)};
    p.strips_ = (width + stripWidth - 1) / stripWidth;
    p.blocks = (height + blockRows - 1) / blockRows;
    p.padded = p.strips_ * stripWidth;
    p.stripStride = size_t(p.blocks) * blockValues;
    if (p.stripStride > maxSize / p.strips_ / sizeof(int16_t) - 64)
        return false;
    if (sigma == 0)
        return true;
#ifdef IIRBLUR_PROFILE
    auto stamp = [](int i) {
        profile[i] = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    };
    stamp(0);
#endif
    const bool parallel = size_t(width) * height >= 16384;
    Executor &executor = sharedExecutor();
    const unsigned slots = parallel ? executor.slots() : 1;
    // Every buffer is allocated before the first pixel changes, so a failed
    // allocation never leaves a half-filtered image behind.
    CallerStorage &storage = callerStorage();
    p.scratchFloats = std::max(size_t(p.padded) * 12, size_t(height) * 3 * stripWidth);
    if (p.scratchFloats > maxSize / sizeof(float) / slots - 64)
        return false;
    p.scratch = aligned64(storage.scratch, p.scratchFloats * slots);
    if (!p.packed)
        p.strips = aligned64(storage.strips, size_t(p.strips_) * p.stripStride);
    executor.run(p.blocks, parallel,
                 [&](unsigned job, unsigned slot) { horizontalTask(p, job, p.scratchFor(slot)); });
#ifdef IIRBLUR_PROFILE
    stamp(1);
#endif
    // Scatter concurrently processed tasks across the image so that no two
    // workers keep touching the same output cache lines at the same time.
    const unsigned tasks = (p.strips_ + taskStrips - 1) / taskStrips;
    const unsigned step = tasks > 32 && tasks % 17 != 0 ? 17 : 1;
    executor.run(tasks, parallel, [&](unsigned job, unsigned slot) {
        verticalTask(p, unsigned((size_t(job) * step) % tasks), p.scratchFor(slot));
    });
#ifdef IIRBLUR_PROFILE
    stamp(2);
#endif
    return true;
} catch (const std::bad_alloc &) {
    return false;
} catch (const std::length_error &) {
    return false;
}
} // namespace iirblur
