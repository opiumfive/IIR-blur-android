#include "../app/jni/IIRBlur.h"
#include "reference.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef IIRBLUR_HAVE_BASELINE
extern "C" void legacy(uint8_t *, unsigned, unsigned, float);
extern "C" void fp16(uint8_t *, unsigned, unsigned, float);
namespace iirblur_prev {
bool blur(uint8_t *, unsigned, unsigned, size_t, float);
}
#endif

static iirblur::Quality testQuality = iirblur::Quality::automatic;

static void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
static uint32_t randomBits(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
static void fill(uint8_t *p, unsigned w, unsigned h, size_t stride, unsigned pattern) {
    uint32_t random = 1234567;
    for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x) {
            for (unsigned c = 0; c < 3; ++c) {
                uint8_t value = randomBits(random);
                if (pattern == 1)
                    value = c == 0 ? 255 : c == 1 ? 127 : 0;
                if (pattern == 2)
                    value = c == 0 ? (x < w / 2 ? 0 : 255) : (y < h / 2 ? 255 : 0);
                if (pattern == 3)
                    value = x == w / 2 && y == h / 2 ? 255 : 0;
                if (pattern == 4)
                    value = ((x / 7 + y / 9) % 2) ? 255 : 0;
                if (pattern == 5)
                    value = (x + c * y) % 256;
                p[y * stride + x * 4 + c] = value;
            }
            p[y * stride + x * 4 + 3] = randomBits(random); // Alpha must never be overwritten.
        }
}
struct Error {
    unsigned max = 0;
    double sum = 0;
    size_t count = 0;
};
static Error checkCase(unsigned w, unsigned h, float sigma, unsigned pattern, unsigned padding) {
    size_t stride = size_t(w) * 4 + padding;
    constexpr size_t guard = 37;
    std::vector<uint8_t> input(guard + stride * h + guard, 0xa5);
    fill(input.data() + guard, w, h, stride, pattern);
    auto expected = input, output = input;
    referenceBlur(expected.data() + guard, w, h, stride, sigma);
    require(iirblur::blur(output.data() + guard, w, h, stride, sigma, testQuality), "blur failed");
    Error error;
    for (size_t i = 0; i < input.size(); ++i) {
        size_t rowByte = i >= guard ? (i - guard) % stride : stride;
        bool rgb =
            i >= guard && i < guard + stride * h && rowByte < size_t(w) * 4 && rowByte % 4 != 3;
        if (rgb) {
            unsigned delta = std::abs(int(output[i]) - expected[i]);
            if (delta > 1 || (pattern == 1 && output[i] != input[i])) {
                std::fprintf(
                    stderr,
                    "FAIL %ux%u sigma=%g pattern=%u padding=%u byte=%zu expected=%u actual=%u\n", w,
                    h, sigma, pattern, padding, i, expected[i], output[i]);
                throw std::runtime_error("reference mismatch");
            }
            error.max = std::max(error.max, delta);
            error.sum += delta;
            ++error.count;
        } else
            require(output[i] == input[i], "alpha, padding or guard overwritten");
    }
    return error;
}
static void concurrencyTest() {
    constexpr unsigned count = 6;
    std::atomic<bool> ok{true};
    std::vector<std::thread> callers;
    for (unsigned i = 0; i < count; ++i)
        callers.emplace_back([&, i] {
            try {
                for (unsigned j = 0; j < 10; ++j)
                    checkCase(129 + i * 3, 131 + j, j % 2 ? 30.f : 100.f, j % 6, 7);
            } catch (...) {
                ok = false;
            }
        });
    for (auto &thread : callers)
        thread.join();
    require(ok, "concurrent calls failed");
}
static void tests() {
    Error total;
    unsigned cases = 0;
    auto test = [&](unsigned w, unsigned h, float s, unsigned p, unsigned pad) {
        auto e = checkCase(w, h, s, p, pad);
        ++cases;
        total.max = std::max(total.max, e.max);
        total.sum += e.sum;
        total.count += e.count;
    };
    for (unsigned w : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u})
        for (unsigned h : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u})
            for (float s : {0.f, .1f, .5f, 3.f, 30.f, 1000.f})
                for (unsigned p = 0; p < 4; ++p)
                    test(w, h, s, p, 13);
    for (unsigned w : {127u, 128u, 129u, 130u, 131u})
        for (unsigned h : {127u, 128u, 129u, 130u, 131u})
            for (unsigned p = 0; p < 6; ++p)
                test(w, h, 30.f, p, 0);
    for (float s : {.5f, 1.f, 3.f, 10.f, 30.f, 60.f, 100.f, 200.f, 500.f, 1000.f})
        for (unsigned p = 0; p < 6; ++p)
            test(641, 519, s, p, 13);
    uint8_t pixel[4] = {0, 0, 0, 255};
    require(!iirblur::blur(nullptr, 1, 1, 4, 1), "null accepted");
    require(!iirblur::blur(pixel, 0, 1, 4, 1), "zero width accepted");
    require(!iirblur::blur(pixel, 1, 0, 4, 1), "zero height accepted");
    require(!iirblur::blur(pixel, 1, 1, 3, 1), "short stride accepted");
    for (float s :
         {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
        require(!iirblur::blur(pixel, 1, 1, 4, s), "invalid sigma accepted");
    require(!iirblur::blur(pixel, ~0u, ~0u, ~size_t(0), 1), "overflow accepted");
    concurrencyTest();
    std::printf("PASS %u reference cases + invalid inputs + 60 concurrent calls; max_error=%u "
                "MAE=%.8f (0..255)\n",
                cases, total.max, total.sum / total.count);
}
// Draft mode is an approximation by design; report its deviation from the
// reference instead of enforcing the one-level bound, and check the invariants
// it must keep: alpha, padding, guards and constant colours.
static void draftReport() {
    for (float s : {12.f, 16.f, 24.f, 30.f, 48.f, 64.f, 100.f})
        for (unsigned p = 0; p < 6; ++p) {
            const unsigned w = 641, h = 519, margin = unsigned(3 * s);
            const size_t stride = size_t(w) * 4 + 13;
            std::vector<uint8_t> input(stride * h, 0xa5);
            fill(input.data(), w, h, stride, p);
            auto expected = input, output = input;
            referenceBlur(expected.data(), w, h, stride, s);
            require(iirblur::blur(output.data(), w, h, stride, s, iirblur::Quality::draft),
                    "draft blur failed");
            unsigned maxAll = 0, maxInterior = 0;
            double sumInterior = 0;
            size_t countInterior = 0;
            for (unsigned y = 0; y < h; ++y) {
                for (unsigned x = 0; x < w; ++x) {
                    bool interior = x >= margin && x + margin < w && y >= margin && y + margin < h;
                    for (unsigned c = 0; c < 3; ++c) {
                        size_t i = y * stride + x * 4 + c;
                        unsigned d = std::abs(int(output[i]) - expected[i]);
                        maxAll = std::max(maxAll, d);
                        if (interior) {
                            maxInterior = std::max(maxInterior, d);
                            sumInterior += d;
                            ++countInterior;
                        }
                        require(p != 1 || output[i] == input[i], "draft changed a constant colour");
                    }
                    require(output[y * stride + x * 4 + 3] == input[y * stride + x * 4 + 3],
                            "draft overwrote alpha");
                }
                for (size_t b = size_t(w) * 4; b < stride; ++b)
                    require(output[y * stride + b] == input[y * stride + b],
                            "draft overwrote row padding");
            }
            std::printf("draft sigma=%g pattern=%u: interior(3 sigma from borders) max_error=%u "
                        "MAE=%.4f; whole image max_error=%u\n",
                        s, p, maxInterior, countInterior ? sumInterior / countInterior : 0., maxAll);
        }
    for (unsigned w : {1u, 5u, 17u, 33u, 64u, 129u})
        for (unsigned h : {1u, 3u, 16u, 31u, 130u})
            for (float s : {12.f, 30.f, 100.f})
                for (unsigned p = 0; p < 3; ++p) {
                    constexpr size_t guard = 37;
                    const size_t stride = size_t(w) * 4 + 7;
                    std::vector<uint8_t> input(guard + stride * h + guard, 0xa5);
                    fill(input.data() + guard, w, h, stride, p);
                    auto output = input;
                    require(iirblur::blur(output.data() + guard, w, h, stride, s,
                                          iirblur::Quality::draft),
                            "draft blur failed");
                    for (size_t i = 0; i < input.size(); ++i) {
                        size_t rowByte = i >= guard ? (i - guard) % stride : stride;
                        bool rgb = i >= guard && i < guard + stride * h &&
                                   rowByte < size_t(w) * 4 && rowByte % 4 != 3;
                        if (!rgb || p == 1)
                            require(output[i] == input[i], "draft: guard, alpha or constant colour changed");
                    }
                }
    std::puts("draft: PASS alpha, padding, guard and constant-colour checks on small sizes");
}
static void fast(uint8_t *p, unsigned w, unsigned h, float sigma) {
    require(iirblur::blur(p, w, h, size_t(w) * 4, sigma, iirblur::Quality::fast), "blur failed");
}
static void precise(uint8_t *p, unsigned w, unsigned h, float sigma) {
    require(iirblur::blur(p, w, h, size_t(w) * 4, sigma, iirblur::Quality::precise), "blur failed");
}
static void draft(uint8_t *p, unsigned w, unsigned h, float sigma) {
    require(iirblur::blur(p, w, h, size_t(w) * 4, sigma, iirblur::Quality::draft), "blur failed");
}
#ifdef IIRBLUR_HAVE_BASELINE
static void previous(uint8_t *p, unsigned w, unsigned h, float sigma) {
    require(iirblur_prev::blur(p, w, h, size_t(w) * 4, sigma), "blur failed");
}
#endif
static std::string onlyVariants; // comma separated names, empty = all
static unsigned gapMs = 0;       // idle time before every call, like a frame interval
static void benchmark(unsigned w, unsigned h, float sigma, unsigned rounds) {
    require(w && h && rounds >= 3 && sigma > 0 && std::isfinite(sigma),
            "invalid benchmark arguments");
    std::vector<uint8_t> input(size_t(w) * h * 4), output(input.size());
    fill(input.data(), w, h, size_t(w) * 4, 0);
    for (size_t i = 3; i < input.size(); i += 4)
        input[i] = 255;
    struct Variant {
        const char *name;
        void (*fn)(uint8_t *, unsigned, unsigned, float);
        std::vector<double> times;
        double cold = 0;
    };
    std::vector<Variant> variants = {{"fast", fast, {}}, {"precise", precise, {}}, {"draft", draft, {}}};
#ifdef IIRBLUR_HAVE_BASELINE
    variants.push_back({"previous", previous, {}});
    variants.push_back({"original_neon", legacy, {}});
    variants.push_back({"original_fp16", fp16, {}});
#endif
    if (!onlyVariants.empty()) {
        std::vector<Variant> selected;
        for (auto &v : variants)
            if (("," + onlyVariants + ",").find("," + std::string(v.name) + ",") != std::string::npos)
                selected.push_back(v);
        require(!selected.empty(), "no variant matches --only");
        variants = selected;
    }
    // Rotate order each round so CPU frequency and temperature do not favour
    // one implementation. Allocation inside the blur is included; reset is not.
    // Note that the memory controller's own frequency scaling reacts to the
    // mix: compute-bound variants let it drop, which slows memory-bound ones
    // measured right after them. --only measures a subset in isolation.
    for (unsigned round = 0; round < rounds + 10; ++round)
        for (unsigned j = 0; j < variants.size(); ++j) {
            auto &v = variants[(j + round) % variants.size()];
            output = input;
            if (gapMs)
                std::this_thread::sleep_for(std::chrono::milliseconds(gapMs));
            auto before = std::chrono::steady_clock::now();
            v.fn(output.data(), w, h, sigma);
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before)
                    .count();
            if (round == 0)
                v.cold = ms;
            if (round >= 10)
                v.times.push_back(ms);
        }
    for (auto &v : variants) {
        std::sort(v.times.begin(), v.times.end());
        std::printf("{\"variant\":\"%s\",\"width\":%u,\"height\":%u,\"sigma\":%g,\"rounds\":%u,"
                    "\"gap_ms\":%u,\"cold_ms\":%.4f,\"median_ms\":%.4f,\"p10_ms\":%.4f,\"p90_ms\":%.4f}\n",
                    v.name, w, h, sigma, rounds, gapMs, v.cold, v.times[rounds / 2],
                    v.times[rounds / 10], v.times[rounds * 9 / 10]);
    }
}
int main(int argc, char **argv) try {
    for (;;) {
        if (argc > 2 && std::string(argv[1]) == "--only") {
            onlyVariants = argv[2];
        } else if (argc > 2 && std::string(argv[1]) == "--gap") {
            gapMs = unsigned(std::stoul(argv[2]));
        } else {
            break;
        }
        argc -= 2;
        argv += 2;
    }
    if (argc > 1 && std::string(argv[1]) == "--test") {
        for (auto q : {iirblur::Quality::fast, iirblur::Quality::precise}) {
            testQuality = q;
            std::printf("%s: ", q == iirblur::Quality::fast ? "fast" : "precise");
            tests();
        }
        draftReport();
    }
    else if (argc > 1 && std::string(argv[1]) == "--concurrency") {
        concurrencyTest();
        std::puts("PASS concurrency");
    } else
        benchmark(argc > 1 ? std::stoul(argv[1]) : 640, argc > 2 ? std::stoul(argv[2]) : 1294,
                  argc > 3 ? std::stof(argv[3]) : 30, argc > 4 ? std::stoul(argv[4]) : 31);
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
