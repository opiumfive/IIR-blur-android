# Reproducing the measurements

The runner builds the production `IIRBlurFast.cpp` directly and executes it on the
connected Android device. It also extracts the original NEON and FP16 sources
from commit `6f1eda35454634912006676eabbbc4388e50044d`. That commit must be present
in the local Git history. Their kernels, per-call allocation, optimizer flags
(`-Os -ffast-math -funroll-loops -fno-strict-aliasing`) and scheduling are retained.
Only the JNI wrapper is removed and their incompatible `ThreadPool` types are
renamed to prevent a C++ ODR violation in the benchmark binary.

The optimized kernel uses `-O3 -ffp-contract=fast`, without `-ffast-math`.
Default toolchain: the project's Android NDK 21.4.7075529, arm64-v8a, API 21,
static libc++. The phone measurements include internal workspace allocation
and synchronization; they exclude input reset and image decoding. Every call
starts with the same deterministic opaque RGBA noise image. Ten warm-up rounds
precede 31 or 41 measured rounds, with variant order rotated each round. Results
include the first call separately, median, p10 and p90. CPU frequency/affinity
and thermal controls are not modified.

```sh
# Uses local.properties or ANDROID_SDK_ROOT; pass --serial if several devices are connected.
python3 benchmarks/run.py --serial YOUR_SERIAL -- --test
python3 benchmarks/run.py --serial YOUR_SERIAL -- 640 1294 30 41
python3 benchmarks/run.py --serial YOUR_SERIAL -- 1080 1920 30 41

# Build once; subsequent runs can execute the same binary directly.
python3 benchmarks/run.py --build-only
adb -s YOUR_SERIAL push build/blur-bench/android/blur-bench /data/local/tmp/iirblur-bench
adb -s YOUR_SERIAL shell /data/local/tmp/iirblur-bench 3840 2160 30 31

# Host memory, arithmetic and race checks; no Android dependencies.
python3 benchmarks/run.py --host --sanitize -- --test
python3 benchmarks/run.py --host --scalar --sanitize -- --test
python3 benchmarks/run.py --host --tsan -- --concurrency

# Bitmap/JNI contract and a benchmark of the bundled back.png, decoded at 640x1294.
./gradlew :app:connectedDebugAndroidTest
adb logcat -d -s IIRBlurBenchmark:I
```

`--ndk PATH` selects another installed NDK; this changes the compiler for both
baseline and optimized sources, so record it when comparing results. `--scalar`
forces the portable production backend. No emulators are used for timing.

## Correctness

The independent reference in `reference.h` evaluates the original third-order
recurrence in double precision, including its four sweep directions and edge
initialization. It does not share the optimized delta recurrence. The tests
cover 3114 combinations of dimensions, sigma and patterns, plus 60 overlapping
calls on distinct buffers and invalid inputs. Cases include 1x1, single rows and
columns, all SIMD tails, misaligned input, padded strides, random alpha, constant
colours, impulses, steps, checkerboards and gradients. Sigma ranges from 0 to
1000. Guards, row padding and alpha must be byte-identical; RGB can differ from
the double reference by at most one level out of 255. Constant colours must be
exactly preserved.

Host NEON and forced scalar runs passed AddressSanitizer and
UndefinedBehaviorSanitizer. The concurrent test passed ThreadSanitizer. The
same correctness suite passed on the Samsung ARM64 CPU. Android instrumentation
also checks the real Bitmap/JNI path, invalid sigma, immutable bitmaps and RGB565.

## Why the implementation is faster

* Four independent rows keep separate recurrences in SIMD registers. Pixel loads
  widen an entire RGBA word instead of assembling vectors from individual bytes.
* The forward horizontal sweep uses a small per-worker row buffer. The reverse
  sweep writes directly into four-column RGB strips, fusing the layout change
  with filtering. Vertical sweeps then stream through contiguous memory.
* Strip height is padded to four rows and the workspace base is aligned to 64
  bytes, so adjacent horizontal tasks do not write the same cache line.
* Vertical lanes contain 12 consecutive RGB channels, including mixed channels
  in each register. No deinterleaving is needed for the recurrence. Output is
  packed into one 16-byte RGBA store, preserving the original alpha.
* Up to seven persistent workers plus the caller claim small tasks dynamically;
  each dispatch has its own completion barrier. This balances heterogeneous
  cores without pinning. Small images stay on the caller.
* Workspaces are reused. Emulated Android TLS is resolved before entering the
  pixel loops; leaving `scratch.data()` inside those loops made NDK 21 emit
  repeated `__emutls_get_address` calls and spill all recurrence registers.
* Coefficients are passed to each kernel by value, so pixel stores cannot force
  the compiler to reload aliased coefficient memory on every iteration.

The main workspace is approximately 12 bytes per pixel (width and height rounded
to four), plus 64 bytes for alignment. Horizontal scratch is 64 bytes per column
per participating thread. These buffers retain their largest capacity until the
owning thread exits. Calls from different threads use separate image workspaces;
parallel phases share the worker pool. Concurrent access to the same input
bitmap still requires synchronization by the caller.

## Numerical stability

Writing the original recurrence as

```
y[n] = B*x[n] + b0*y[n-1] + b1*y[n-2] + b2*y[n-3]
```

and keeping first and second differences gives the equivalent realization

```
acceleration = b2*acceleration + (-1-b1-2*b2)*velocity + B*(x-value)
velocity += acceleration
value += velocity
```

At each edge, `value` is initialized to the edge sample and both differences to
zero. Coefficients are calculated in double precision. `B` and the velocity
coefficient are evaluated as polynomial numerators directly, avoiding subtraction
of almost equal values. The small decimal residuals in the original coefficients
(`0.00001*q*q` and `0.000005*q*q*q`) are retained. This preserves the original
transfer function while greatly reducing FP32 cancellation. For tiny sigma the
approximation's negative `q` is clamped to zero; sigma zero is an immediate no-op.
The four sweeps and their boundary convention otherwise remain unchanged.

These coefficients describe the Young–van Vliet recursive approximation; see the
[recursive Gaussian equations in this research paper](https://link.springer.com/article/10.1186/1687-5281-2014-33).
The filter is an approximation to Gaussian convolution, with O(width*height)
work independent of sigma. There is no downsampling, radius-dependent truncation,
FP16 recurrence, or switch to a box filter in the optimized path.

## Samsung results

Measured on SM-G991B (Galaxy S21), Exynos 2100, Android 15. Raw results and the
measurement conditions are in `samsung-s21.jsonl` and `samsung-s21-device.txt`.
Timing varies with the phone's scheduler and CPU frequency; use medians and
ranges, not an isolated fastest iteration. The FP16 baseline has lower numerical
accuracy, and the original FP32 recurrence can produce large errors at high sigma.

Median times in milliseconds, sigma=30:

| Image | Original NEON | Original FP16 | Optimized | Speedup vs NEON |
|---|---:|---:|---:|---:|
| 128×128 | 0.81 | 0.71 | 0.32 | 2.55× |
| 320×647 | 10.59 | 8.30 | 2.51 | 4.22× |
| 640×1294 | 23.53 | 17.35 | 8.53 | 2.76× |
| 1080×1920 | 29.07 | 29.88 | 12.20 | 2.38× |
| 1920×1080 | 24.08 | 24.75 | 11.25 | 2.14× |
| 3840×2160 | 73.69 | 78.40 | 32.48 | 2.27× |

In a separate final-APK instrumentation run on the bundled photo, the actual
Bitmap/JNI call took **2.64 ms median**, p10 **2.33 ms**, p90 **3.03 ms**, at
640×1294 and sigma=30. See `samsung-s21-jni.txt`. This run repeatedly invokes only
the optimized method inside the Android app. Its source image, cache behaviour
and scheduling conditions differ from the alternating native A/B benchmark;
its absolute time must not be divided into the baseline times in the table.
All three instrumentation tests passed. Debug and release builds succeeded for
arm64-v8a, armeabi-v7a, x86 and x86_64; release lint-vital and the existing JVM test
also passed. Performance was measured only on the Samsung ARM64 device.

The demo Activity preserves the original resource-density scaling. On this
Samsung, its RUN button therefore processes **1920×3882**, even though back.png
is stored at 640×1294. Its one-shot, sequential UI timings are not the warm
benchmark medians above. The final APK's RUN flow was also exercised successfully
and its rendered blur was visually inspected.
