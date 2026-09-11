#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Independent double-precision direct-form implementation of the original
// four sweeps and edge initialization. Does not use the optimized delta form.
inline void referenceBlur(uint8_t *image, unsigned w, unsigned h, size_t stride, double sigma) {
    if (sigma == 0)
        return;
    double q =
        sigma >= 2.5 ? .98711 * sigma - .96330 : 3.97156 - 4.14554 * std::sqrt(1 - .26891 * sigma);
    q = std::max(0., q);
    double d = 1.57825 + 2.44413 * q + 1.4281 * q * q + .422205 * q * q * q;
    double a = (2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q) / d;
    double b = -(1.4281 * q * q + 1.26661 * q * q * q) / d;
    double c = .422205 * q * q * q / d;
    double gain = 1 - (a + b + c);
    std::vector<double> buffer(size_t(w) * h * 3);
    auto sweep = [&](size_t start, ptrdiff_t step, unsigned count, auto input, auto output) {
        double p = input(start), p1 = p, p2 = p;
        for (unsigned i = 0; i < count; ++i) {
            size_t index = static_cast<size_t>(static_cast<ptrdiff_t>(start) + step * i);
            double v = gain * input(index) + a * p + b * p1 + c * p2;
            output(index, v);
            p2 = p1;
            p1 = p;
            p = v;
        }
    };
    auto read = [&](size_t i) { return buffer[i]; };
    auto write = [&](size_t i, double v) { buffer[i] = v; };
    for (unsigned y = 0; y < h; ++y)
        for (unsigned channel = 0; channel < 3; ++channel) {
            size_t first = size_t(y) * w * 3 + channel;
            sweep(
                first, 3, w,
                [&](size_t i) {
                    return double(image[(i / 3 / w) * stride + (i / 3 % w) * 4 + i % 3]);
                },
                write);
            sweep(first + size_t(w - 1) * 3, -3, w, read, write);
        }
    for (unsigned x = 0; x < w; ++x)
        for (unsigned channel = 0; channel < 3; ++channel) {
            size_t first = x * 3 + channel;
            sweep(first + size_t(h - 1) * w * 3, -ptrdiff_t(w) * 3, h, read, write);
            sweep(first, ptrdiff_t(w) * 3, h, read, [&](size_t i, double v) {
                image[(i / 3 / w) * stride + (i / 3 % w) * 4 + i % 3] =
                    static_cast<uint8_t>(std::max(0., std::min(255., v + .5)));
            });
        }
}
