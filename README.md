# IIR Blur (Android, NDK)

CPU Gaussian blur approximation with O(width × height) work, independent of the
blur radius. `Utils.blurNeon` uses an optimized ARM64 NEON implementation and a
portable fallback on other Android ABIs. The scalar, FP16, box and GPU versions
remain in the demo for comparison.

```java
Bitmap bitmap = source.copy(Bitmap.Config.ARGB_8888, true);
Utils.blurNeon(bitmap, 30f);                             // automatic quality
Utils.blurNeon(bitmap, 30f, Utils.QUALITY_DRAFT);        // fastest, approximate
Utils.setBlurIdleSpin(40);                               // e.g. blur every 30 ms frame
```

The bitmap must be mutable, software-backed ARGB_8888. RGB is blurred in place;
alpha is preserved. The implementation respects row stride and handles arbitrary
image sizes. Sigma must be finite and non-negative; zero is a no-op.

Three quality levels (`iirblur::Quality` in C++, `Utils.QUALITY_*` in Java):

| Quality | What it does | Accuracy against the double reference |
|---|---|---|
| `fast` | Keeps the horizontally filtered image as 8-bit RGB in the bitmap itself: no image-sized buffer, least memory traffic. | Within one level for sigma up to 128; mean error 0.035 levels. |
| `precise` | 16-bit fixed point intermediate in a separate buffer (6 bytes per pixel). Any sigma. | Within one level; mean error 0.003 levels. |
| `draft` | Blurs a box-downsampled copy (2x, 4x or 8x, from sigma 12) with the fast kernel and interpolates it back. | Interior within 4 levels; mean under 0.5 level on noise and edges, up to 1.8 on hard high-contrast seams; up to 25 levels at the borders, where the boundary convention differs. |
| `automatic` | `fast` up to sigma 128, `precise` beyond. | As above. |

## Performance

Samsung Galaxy S21 (Exynos 2100), sigma 30, medians of 41 rounds, native
benchmark. "Original" is the NEON implementation this project started from;
"previous" is the first optimized version. Details, conditions and the
reproducible raw data are in [the benchmark report](benchmarks/README.md).

| Image | Original NEON | Previous | Fast | Precise | Draft | Fast, back to back | Draft, back to back |
|---|---:|---:|---:|---:|---:|---:|---:|
| 320x647 | 2.97 | 1.79 | 0.30 | 0.29 | 0.18 | 0.31 | 0.18 |
| 640x1294 | 10.46 | 6.87 | 1.12 | 1.14 | 0.44 | 1.11 | 0.46 |
| 1080x1920 | 24.70 | 18.71 | 4.30 | 3.36 | 1.25 | 3.04 | 1.26 |
| 1920x1080 | 21.80 | 20.22 | 3.75 | 3.19 | 1.25 | 2.94 | 1.27 |
| 3840x2160 | 80.14 | 35.66 | 14.09 | 13.56 | 5.34 | 13.66 | 5.77 |

With a 16 ms idle gap before every call (a blur per frame at 60 Hz), 1080x1920:
original 32.4 ms, previous 14.6 ms, fast 3.0 ms,
draft 1.2 ms. The first columns come from a run where all
variants alternate; "back to back" repeats only the new kernels, which is what
an animation sees. Medians in milliseconds.

Two things matter as much as the kernel on a phone:

* **Clocks.** CPU and memory clocks drop within a few milliseconds of idling
  and take longer than a blur to ramp up, so a blur once per frame ran three
  times slower than back to back. The worker threads therefore stay runnable
  for 20 ms after a call, waiting in `wfe` at idle scheduling priority (almost
  no power, no competition with other threads, but the cluster keeps its
  clock). `setBlurIdleSpin(ms)` sets that duration; 0 disables it.
* **First call.** The first blur in a process pays for thread creation, page
  faults and cold clocks: 5-15 ms at 1080x1920 rather than 3.

See [benchmarks, correctness checks and implementation details](benchmarks/README.md)
for the reproducible comparisons against the original source on a Samsung
Galaxy S21, and for what changed in the implementation.

## Try the demo

[Download the updated demo APK](demo.apk).

The demo blurs the bundled photo, scaled by the display density to 1920×3882
on this Samsung, with every method and shows the time of a first (cold) and a
second call. These are one-shot UI timings; the repeatable measurements are in
the benchmark report.

![Optimized CPU blur demo on Samsung Galaxy S21](demo.png)

## Build and test

Android SDK 34, NDK 21.4.7075529, JDK 17+:

```sh
./gradlew :app:assembleDebug
./gradlew :app:connectedDebugAndroidTest
python3 benchmarks/run.py -- --test
```

The demo APK is built at `app/build/outputs/apk/debug/app-debug.apk`.
