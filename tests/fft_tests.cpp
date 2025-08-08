#include <algorithm>
#include <array>
#include <format>
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "fft.h"

TEST_CASE("FFT")
{
    constexpr std::array kFFTSize = {32, 64, 128, 256, 512, 1024};

    for (auto fft_size : kFFTSize)
    {
        auto subcase_name = std::format("FFT size: {}", fft_size);
        SECTION(subcase_name.c_str())
        {
            sfFDN::FFT fft(fft_size);

            auto input_buffer = fft.AllocateRealBuffer();
            auto output_buffer = fft.AllocateComplexBuffer();

            // Fill with white noise
            std::vector<float> expected_buffer(input_buffer.Data().size(), 0.f);
            std::default_random_engine generator;
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto i = 0; i < input_buffer.Data().size(); ++i)
            {
                input_buffer.Data()[i] = dist(generator);
                expected_buffer[i] = input_buffer.Data()[i];
            }

            std::ranges::fill(output_buffer.Data(), 0.f);

            fft.Forward(input_buffer, output_buffer);
            fft.Inverse(output_buffer, input_buffer);

            const float scale = 1.0f / static_cast<float>(fft_size);
            for (auto i = 0; i < input_buffer.Data().size(); ++i)
            {
                REQUIRE_THAT(input_buffer.Data()[i] * scale, Catch::Matchers::WithinAbs(expected_buffer[i], 1e-4));
            }
        }
    }
}