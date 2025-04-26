#include "doctest.h"
#include "nanobench.h"

#include <array>
#include <iostream>
#include <random>

#include <parallel_gains.h>

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("ParallelGainsPerf")
{
    constexpr size_t SR = 48000;
    constexpr size_t block_size = 512;
    constexpr size_t ITER = ((SR / block_size) + 1); // 1 second at 48kHz
    constexpr size_t N = 16;
    constexpr std::array kGains = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                                   0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    std::cout << "ITER: " << ITER << std::endl;
    std::cout << "BLOCK SIZE: " << block_size << std::endl;
    std::cout << "N: " << N << std::endl;

    std::vector<float> input(block_size, 0.f);
    std::vector<float> output(block_size * N, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    nanobench::Bench bench;
    bench.title("FDN Perf");
    bench.minEpochIterations(100);
    bench.timeUnit(1us, "us");
    bench.relative(true);

    bench.run("ParallelGains - Input", [&] {
        fdn::ParallelGains input_gains;

        input_gains.SetGains(kGains);

        for (size_t i = 0; i < ITER; ++i)
        {
            input_gains.ProcessBlock(input, output);
        }
    });

    bench.run("ParallelGains - Output", [&] {
        fdn::ParallelGains output_gains;

        output_gains.SetGains(kGains);

        for (size_t i = 0; i < ITER; ++i)
        {
            output_gains.ProcessBlock(output, input);
        }
    });
}
