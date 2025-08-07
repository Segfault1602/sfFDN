#include "doctest.h"
#include "nanobench.h"

#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>

#include "sffdn/sffdn.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_SUITE_BEGIN("ParallelGains");

TEST_CASE("ParallelInputGainsPerf")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t N = 16;
    constexpr std::array kGains = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                                   0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize * N, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    nanobench::Bench bench;
    bench.title("Parallel Input Gain Perf");
    bench.minEpochIterations(100);
    // bench.batch(kBlockSize);

    sfFDN::ParallelGains input_gains(sfFDN::ParallelGainsMode::Multiplexed);
    input_gains.SetGains(kGains);
    bench.run("ParallelGains - Input", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input.data());
        sfFDN::AudioBuffer output_buffer(kBlockSize, N, output.data());
        input_gains.Process(input_buffer, output_buffer);
    });

    std::filesystem::path output_dir = std::filesystem::current_path() / "perf";
    if (!std::filesystem::exists(output_dir))
    {
        std::filesystem::create_directory(output_dir);
    }
    std::filesystem::path filepath = output_dir / std::format("parallel_input_gain_B{}.json", kBlockSize);
    std::ofstream render_out(filepath);
    bench.render(ankerl::nanobench::templates::json(), render_out);
}

TEST_CASE("ParallelOutputGainsPerf")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t N = 16;
    constexpr std::array kGains = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                                   0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize * N, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    nanobench::Bench bench;
    bench.title("Parallel Output Gain Perf");
    bench.minEpochIterations(100);
    // bench.batch(kBlockSize);

    sfFDN::ParallelGains output_gains(sfFDN::ParallelGainsMode::DeMultiplexed);
    output_gains.SetGains(kGains);
    bench.run("ParallelGains - Output", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, N, input.data());
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output.data());
        output_gains.Process(input_buffer, output_buffer);
    });

    std::filesystem::path output_dir = std::filesystem::current_path() / "perf";
    if (!std::filesystem::exists(output_dir))
    {
        std::filesystem::create_directory(output_dir);
    }
    std::filesystem::path filepath = output_dir / std::format("parallel_output_gain_B{}.json", kBlockSize);
    std::ofstream render_out(filepath);
    bench.render(ankerl::nanobench::templates::json(), render_out);
}

TEST_SUITE_END();