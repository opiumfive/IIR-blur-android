# Reproducing the measurements

The runner builds the production `IIRBlurFast.cpp` directly and executes it on the
connected Android device. It also extracts the original NEON and FP16 sources
from commit `6f1eda35454634912006676eabbbc4388e50044d` and the first optimized
core from commit `455f960` ("previous"). Those commits must be present in the
local Git history. The originals keep their kernels, per-call allocation,
optimizer flags (`-Os -ffast-math -funroll-loops -fno-strict-aliasing`) and
scheduling; only the JNI wrapper is removed and their colliding `ThreadPool`
types are renamed. "Previous" is compiled unchanged under another namespace.

The current kernel uses `-O3 -ffp-contract=fast`, without `-ffast-math`.
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
python3 benchmarks/run.py --serial YOUR_SERIAL -- 1080 1920 30 41
# Only some variants, back to back:
python3 benchmarks/run.py --serial YOUR_SERIAL -- --only fast,precise,draft 1080 1920 30 41
# Sleep 16 ms before every call, like a blur once per frame:
python3 benchmarks/run.py --serial YOUR_SERIAL -- --only fast,draft --gap 16 1080 1920 30 41

# Build once; subsequent runs can execute the same binary directly.
python3 benchmarks/run.py --build-only
adb -s YOUR_SERIAL push build/blur-bench/android/blur-bench /data/local/tmp/iirblur-bench
adb -s YOUR_SERIAL shell /data/local/tmp/iirblur-bench 3840 2160 30 31

# Host memory, arithmetic and race checks; no Android dependencies.
python3 benchmarks/run.py --host --sanitize -- --test
python3 benchmarks/run.py --host --scalar --sanitize -- --test
python3 benchmarks/run.py --host --tsan -- --concurrency
# Executor stress: 3000 short parallel blurs from four callers with the
# workers alternately asleep and awake; a watchdog fails on a lost wake-up.
python3 benchmarks/run.py --host --tsan -- --stress

# Markdown tables from a results file.
python3 benchmarks/tables.py benchmarks/samsung-s21.jsonl

# Bitmap/JNI contract and a benchmark of the bundled back.png, decoded at 640x1294.
./gradlew :app:connectedDebugAndroidTest
adb logcat -d -s IIRBlurBenchmark:I
```

`--ndk PATH` selects another installed NDK; this changes the compiler for all
sources, so record it when comparing results. `--scalar` forces the portable
production backend. No emulators are used for timing.

## Correctness

The independent reference in `reference.h` evaluates the original third-order
recurrence in double precision, including its four sweep directions and edge
initialization. It does not share the optimized delta recurrence. The tests
cover 3114 combinations of dimensions, sigma and patterns, plus 60 overlapping
calls on distinct buffers and invalid inputs, once for the fast and once for the
precise quality. Cases include 1x1, single rows and columns, all SIMD tails,
misaligned input, padded strides, random alpha, constant colours, impulses,
steps, checkerboards and gradients. Sigma ranges from 0 to 1000. Guards, row
padding and alpha must be byte-identical; RGB can differ from the double
reference by at most one level out of 255. Constant colours must be exactly
preserved.

Draft quality is an approximation by design. The test suite reports its
deviation from the reference for each pattern and sigma, separately for the
interior (three sigma away from the borders) and for the whole image, and checks
that alpha, padding, guards and constant colours are untouched at 90 small
sizes. See "Draft quality" below for the numbers.

Host NEON and forced scalar runs passed AddressSanitizer and
UndefinedBehaviorSanitizer. The concurrent test and the executor stress test
passed ThreadSanitizer; the stress test also runs with the dispatcher's
critical windows widened by injected delays (job publication, worker join,
back-out), which crashed or hung the executor before the fixes it guards.
The same correctness suite and the stress test passed on the Samsung ARM64
CPU. Android instrumentation
also checks the real Bitmap/JNI path for every quality, invalid sigma and
quality values, immutable bitmaps and RGB565.

Mean absolute error against the double reference over the whole suite: 0.0033
levels for precise, 0.035 for fast (the 8-bit intermediate rounds to half a
level, which the vertical pass averages out).

## What makes it fast

The work per pixel is fixed: four sweeps of a third-order recurrence per
channel. Measured on the phone, one recurrence step on a 4-lane vector costs
2.6 cycles on the Cortex-X1 and 4.0 on a Cortex-A78 when six independent
chains are interleaved, so the whole filter is about 30 SIMD instructions per
pixel. What decides the time is everything around it: memory traffic, the
number of independent chains, thread placement and clock frequency.

* **Memory traffic is the ceiling.** All cores together read or write about
  20 GB/s of DRAM on this phone. The previous implementation moved 48 bytes per
  pixel through DRAM (an fp32 intermediate written, rewritten and read again);
  for 1080x1920 that alone is 5 ms. Fast quality keeps the horizontally
  filtered image as 8-bit RGB inside the bitmap: two passes over the pixels,
  16 bytes per pixel and no image-sized buffer at all. Precise quality stores
  a 16-bit fixed point intermediate (6 bytes per pixel) in a planar strip
  layout where a horizontal task writes whole cache lines and a vertical task
  streams through contiguous memory.
* **Rows in lanes.** The horizontal pass filters four rows at once with one
  recurrence per channel, so three chains and no wasted alpha lane; a 4x4
  transpose plus a table lookup turns four RGBA pixels into per-channel
  vectors, with the fixed point scale folded into the conversion instruction.
  The vertical pass keeps eight columns (six chains) in registers.
* **The vertical scratch stays in L2.** The reverse vertical sweep writes to a
  per-thread buffer instead of back to memory; the forward sweep reads it from
  cache and writes the final pixels once, with the original alpha merged in.
* **Nothing goes through memory that should not.** Coefficients travel in one
  vector register (`fmla` by lane), the recurrence states are locals, results
  are narrowed as soon as they exist, and the kernels contain no calls. NDK 21
  turned an inner lambda into a call with its captured state in memory, which
  cost 20% on big cores and 4x on little cores.
* **Explicit prefetch for in-order cores.** A Cortex-A55 stalls on every cache
  miss. Without software prefetch of the four row streams its horizontal pass
  took 100 cycles per pixel, with it 25. Vertical passes prefetch the strided
  bitmap rows sixteen rows ahead.
* **Thread placement.** Workers are bound to their core class (big or little,
  from `cpu_capacity` in sysfs); otherwise the scheduler places a worker that
  has just woken up on a little core and migrates it only after the blur is
  over. Little cores stop claiming tasks while the big cores still have enough
  work left to outlast a little task, so no straggler delays the barrier.
  Concurrently processed vertical tasks are spread across the image so that
  two cores never keep writing the same cache lines.
* **Clocks.** Phone CPUs and the memory controller drop their clocks within a
  few milliseconds of idling, and ramp up slower than a blur lasts; a blur
  called once per frame ran three times slower than back to back. Workers now
  stay runnable for 20 ms after a job, waiting in `wfe` under the `SCHED_IDLE`
  policy (almost no power, no competition with the app's own threads, but the
  cluster keeps its clock), then sleep. `setIdleSpin()` changes the duration;
  0 disables it: workers then block right after a job and every dispatch pays
  the wake-up. A job is closed as soon as the caller runs out of tasks, so a
  worker that is still asleep or starved never delays completion; the last
  worker to leave, whether it worked or backed out of a closed job, wakes the
  caller.
* **Draft quality** blurs a box-downsampled copy (2x, 4x or 8x, chosen so that
  the reconstruction error stays below about a tenth of a level on smooth
  content) with the fast kernel and interpolates it back bilinearly in 16-bit
  fixed point, in one pass that also restores the original alpha. The extra
  variance of the box and the tent is subtracted from the small blur's sigma.
  Opaque images are detected during downsampling and written back without
  reading.

The horizontal pass is now compute bound on the big cores and the vertical
pass is close to it; parallel efficiency over the four big cores is 98%. The
remaining gap to the theoretical minimum is the latency of the recurrence on
the Cortex-X1 (three chains in the horizontal pass) and the little cores.

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

The horizontal recurrence runs on pixel values times 64 (exact in binary
floating point), so the 16-bit intermediate of precise quality is a plain
rounding, and the 8-bit intermediate of fast quality rounds to the nearest
level with saturation. The recursive filter overshoots at hard edges once
sigma is large (0.02 levels at sigma 140, 0.7 at 180, 12 at 1000); 8 bits would
clip that, so automatic quality switches to precise above sigma 128.

These coefficients describe the Young–van Vliet recursive approximation; see the
[recursive Gaussian equations in this research paper](https://link.springer.com/article/10.1186/1687-5281-2014-33).

## Draft quality

Interior of a 641x519 image, three sigma from the borders, against the double
reference (levels out of 255). Patterns: random noise, step edges, a 7x9
checkerboard and wrapped gradients with hard 255-to-0 seams.

| sigma | factor | interior max error | interior mean error | whole-image max error |
|---:|---:|---:|---:|---:|
| 12 | 2 | 2 | 0.11-0.67 | 17 |
| 16 | 2 | 2 | 0.13-0.70 | 16 |
| 24 | 4 | 3 | 0.07-1.17 | 25 |
| 30 | 4 | 3 | 0.06-1.11 | 24 |
| 48 | 8 | 4 | 0.00-1.81 | 25 |
| 64 | 8 | 3 | 0.00-1.84 | 25 |

(At sigma 100 the three-sigma margin leaves no interior at this size; the
whole-image maximum is 25 there as well.) The largest interior means belong to
the wrapped gradients, whose hard seams alias in the box downsampling; the
noise and step patterns stay under 0.5. For comparison, the exact recursive
filter itself differs from a true Gaussian by up to 2.5 levels on the same
patterns. Near the borders draft
quality follows a different boundary convention (block averages instead of
the filter's edge state) and can differ from the reference by 10-25 levels on
noise or at a seam that touches the border; on photographs the difference is
a few levels within two sigma of the border. The fine checkerboard shows the
expected aliasing of box downsampling: a low-amplitude ripple of a few levels.

## Samsung results

Measured on SM-G991B (Galaxy S21), Exynos 2100 (1x Cortex-X1, 3x A78, 4x A55),
Android 15. Raw results and the measurement conditions are in
`samsung-s21.jsonl` and `samsung-s21-device.txt`. Timing varies with the
phone's scheduler and clock governors; use medians and ranges, not an isolated
fastest iteration. The FP16 baseline has lower numerical accuracy, and the
original FP32 recurrence can produce large errors at high sigma.

Two conditions are reported. In the rotated run every variant follows the
others in turn; the originals are compute bound and let the memory controller
clock down, which penalizes the memory-bound new kernels measured right after
them. The back-to-back run measures the new kernels alone, which is what a
blur per frame sees. The gap run sleeps before every call.

Rotated with the older implementations (median ms):

| Image | sigma | Original NEON | Original FP16 | Previous | Fast | Precise | Draft | Fast vs original | Draft vs original |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128x128 | 30 | 0.22 | 0.26 | 0.19 | 0.03 | 0.03 | 0.02 | 6.9x | 11.2x |
| 320x647 | 30 | 2.97 | 2.22 | 1.79 | 0.30 | 0.29 | 0.18 | 10.0x | 16.7x |
| 640x1294 | 30 | 10.46 | 9.56 | 6.87 | 1.12 | 1.14 | 0.44 | 9.4x | 23.9x |
| 1080x1920 | 30 | 24.70 | 26.63 | 18.71 | 4.30 | 3.36 | 1.25 | 5.7x | 19.7x |
| 1920x1080 | 30 | 21.80 | 24.56 | 20.22 | 3.75 | 3.19 | 1.25 | 5.8x | 17.4x |
| 3840x2160 | 30 | 80.14 | 82.96 | 35.66 | 14.09 | 13.56 | 5.34 | 5.7x | 15.0x |
| 640x1294 | 1 | 13.29 | 11.93 | 7.50 | 1.48 | 1.28 | 1.20 | 9.0x | 11.1x |
| 640x1294 | 100 | 12.85 | 10.87 | 7.28 | 1.47 | 1.36 | 0.45 | 8.7x | 28.7x |
| 1920x3882 | 30 | 72.83 | 71.07 | 35.02 | 14.37 | 13.25 | 5.03 | 5.1x | 14.5x |

Back to back, new implementation only (median ms, p90 in brackets):

| Image | sigma | Fast | Precise | Draft |
|---|---:|---:|---:|---:|
| 128x128 | 30 | 0.09 (0.09) | 0.09 (0.09) | 0.04 (0.04) |
| 320x647 | 30 | 0.31 (0.31) | 0.30 (0.31) | 0.18 (0.19) |
| 640x1294 | 30 | 1.11 (1.15) | 1.17 (1.20) | 0.46 (0.47) |
| 1080x1920 | 30 | 3.04 (3.13) | 3.24 (3.36) | 1.26 (1.31) |
| 1920x1080 | 30 | 2.94 (3.11) | 3.19 (3.47) | 1.27 (1.42) |
| 3840x2160 | 30 | 13.66 (14.10) | 14.86 (16.12) | 5.77 (6.06) |
| 640x1294 | 1 | 1.23 (1.32) | 1.28 (1.32) | 1.23 (1.31) |
| 640x1294 | 100 | 1.19 (1.23) | 1.23 (1.29) | 0.45 (0.48) |
| 1920x3882 | 30 | 13.47 (14.25) | 14.28 (14.78) | 5.39 (5.57) |

Idle gap before every call, 1080x1920, sigma 30 (median ms):

| Gap | Original NEON | Previous | Fast | Precise | Draft |
|---:|---:|---:|---:|---:|---:|
| 16 ms | 32.40 | 14.61 | 2.96 | 3.14 | 1.25 |
| 50 ms | 41.76 | 22.40 | 4.52 | 4.36 | 1.84 |

In the final-APK instrumentation run on the bundled photo (640x1294, sigma 30)
the Bitmap/JNI call took 1.16 ms median for fast (p10 1.14, p90 1.40), 1.14 ms for precise and 0.50 ms for draft (p90 2.2); the previous implementation measured 2.64 ms in the same test. This run repeatedly invokes one quality
inside the Android app; its source image, cache behaviour and scheduling
conditions differ from the native benchmark.

The demo Activity preserves the original resource-density scaling. On this
Samsung, its RUN button therefore processes 1920x3882, even though back.png is
stored at 640x1294. It shows a first (cold) call and a second call for each
method.
