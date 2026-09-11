#pragma once

#include <cstddef>
#include <cstdint>

namespace iirblur {
// In-place RGBA8888; stride is in bytes. Alpha is preserved.
// Sigma must be finite and non-negative. Returns false for invalid input or
// allocation failure. Distinct buffers may be processed by concurrent callers.
bool blur(uint8_t *pixels, unsigned width, unsigned height, size_t stride, float sigma);
} // namespace iirblur
