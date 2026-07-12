// Дискретное преобразование Фурье: наивное O(N^2) и БПФ Кули—Тьюки O(N log N).
// Обе функции считают X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N).
#pragma once

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace siglab {

using cd = std::complex<double>;
constexpr double TAU = 6.283185307179586476925286766559;

// Наивное ДПФ прямо по определению — эталон для проверки БПФ.
inline std::vector<cd> dft_naive(const std::vector<cd>& x) {
    const std::size_t n = x.size();
    std::vector<cd> out(n);
    for (std::size_t k = 0; k < n; ++k) {
        cd acc{0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) {
            const double ph = -TAU * static_cast<double>(k) * static_cast<double>(j)
                              / static_cast<double>(n);
            acc += x[j] * cd{std::cos(ph), std::sin(ph)};
        }
        out[k] = acc;
    }
    return out;
}

// Итеративное БПФ по основанию 2 с бит-реверсивной перестановкой.
// Требует N = 2^m.
inline std::vector<cd> fft_radix2(std::vector<cd> x) {
    const std::size_t n = x.size();
    if (n == 0 || (n & (n - 1)) != 0)
        throw std::invalid_argument("fft_radix2: размер должен быть степенью двойки");

    // бит-реверсивная перестановка
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }

    // "бабочки": длина блока удваивается на каждом уровне
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = -TAU / static_cast<double>(len);
        const cd wlen{std::cos(ang), std::sin(ang)};
        for (std::size_t i = 0; i < n; i += len) {
            cd w{1.0, 0.0};
            for (std::size_t j = 0; j < len / 2; ++j) {
                const cd u = x[i + j];
                const cd v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    return x;
}

} // namespace siglab
