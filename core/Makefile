CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

fft_bench: main.cpp fft.hpp
	$(CXX) $(CXXFLAGS) main.cpp -o fft_bench

run: fft_bench
	./fft_bench

clean:
	rm -f fft_bench

.PHONY: run clean
