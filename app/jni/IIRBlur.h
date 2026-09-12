#pragma once

#include <cstddef>
#include <cstdint>

namespace iirblur {
// How the horizontally filtered intermediate is kept between the two passes.
//   fast:      8-bit RGB in the bitmap itself; no extra image-sized buffer,
//              least memory traffic. Within one level of the reference for
//              sigma up to about 128; larger sigmas clip the filter overshoot.
//   precise:   16-bit fixed point in a separate buffer (6 bytes per pixel);
//              valid for any sigma.
//   automatic: fast up to sigma 128, precise beyond.
//   draft:     blurs a box-downsampled copy (2x, 4x or 8x, chosen by sigma;
//              from sigma 12) and interpolates it back: the fastest for large
//              sigma, within about a quarter level on smooth content, with
//              mild aliasing on high-contrast fine textures. Below sigma 12
//              it is the same as fast.
enum class Quality { automatic, fast, precise, draft };

// In-place RGBA8888; stride is in bytes. Alpha is preserved.
// Sigma must be finite and non-negative. Returns false for invalid input or
// allocation failure. Distinct buffers may be processed by concurrent callers.
bool blur(uint8_t *pixels, unsigned width, unsigned height, size_t stride, float sigma,
          Quality quality = Quality::automatic);

// How long the worker threads stay awake (waiting in WFE, i.e. runnable but
// nearly idle) after a blur, default 20 ms. Phone CPUs and memory controllers
// drop their clocks within milliseconds of idling and ramp up slower than a
// blur lasts, so a blur called once per frame would otherwise run 3-5x slower
// than back to back. Set it above the frame interval for animations, or 0 for
// one-shot use where nothing should stay awake.
void setIdleSpin(unsigned milliseconds);
} // namespace iirblur
