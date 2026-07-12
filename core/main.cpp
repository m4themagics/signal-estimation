// Самопроверка и бенчмарк: наивное ДПФ против БПФ.
//   1. На случайном сигнале сравниваем результаты обоих алгоритмов.
//   2. Проверяем аналитический случай: спектр чистой синусоиды.
//   3. Меряем время и показываем, как расходятся O(N^2) и O(N log N).
#include "fft.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using siglab::cd;

namespace {

double max_diff(const std::vector<cd>& a, const std::vector<cd>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

std::vector<cd> random_signal(std::size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<cd> x(n);
    for (auto& v : x) v = cd{dist(gen), 0.0};
    return x;
}

template <typename F>
double time_ms(F&& f, int reps) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

} // namespace

int main() {
    // --- проверка 1: БПФ совпадает с определением ---
    const auto x = random_signal(1024, 42);
    const double err = max_diff(siglab::dft_naive(x), siglab::fft_radix2(x));
    std::printf("Проверка на случайном сигнале (N=1024): max|DFT-FFT| = %.3e  %s\n",
                err, err < 1e-9 ? "OK" : "FAIL");

    // --- проверка 2: sin(2*pi*k0*n/N) даёт пики ±N/2 в бинах k0 и N-k0 ---
    const std::size_t n = 256, k0 = 17;
    std::vector<cd> s(n);
    for (std::size_t i = 0; i < n; ++i)
        s[i] = std::sin(siglab::TAU * static_cast<double>(k0 * i) / static_cast<double>(n));
    const auto sp = siglab::fft_radix2(s);
    const double peak_err = std::max(std::abs(sp[k0] - cd{0.0, -static_cast<double>(n) / 2}),
                                     std::abs(sp[n - k0] - cd{0.0, static_cast<double>(n) / 2}));
    std::printf("Проверка на синусоиде (N=256, k0=17): ошибка пиков = %.3e  %s\n\n",
                peak_err, peak_err < 1e-9 ? "OK" : "FAIL");

    // --- бенчмарк ---
    std::printf("%8s %14s %14s %10s\n", "N", "наивное, мс", "БПФ, мс", "ускорение");
    for (std::size_t sz : {256, 1024, 4096, 16384}) {
        const auto sig = random_signal(sz, 7);
        const int reps_naive = sz <= 1024 ? 20 : 3;
        const double t_naive = time_ms([&] { siglab::dft_naive(sig); }, reps_naive);
        const double t_fft = time_ms([&] { siglab::fft_radix2(sig); }, 200);
        std::printf("%8zu %14.3f %14.4f %9.0fx\n", sz, t_naive, t_fft, t_naive / t_fft);
    }
    return err < 1e-9 && peak_err < 1e-9 ? 0 : 1;
}
