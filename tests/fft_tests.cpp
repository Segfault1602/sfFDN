#include "doctest.h"

#include <array>
#include <format>
#include <random>

#include "fft.h"

TEST_CASE("FFT")
{
    constexpr std::array kFFTSize = {32, 64, 128, 256, 512, 1024};

    for (size_t fft_size : kFFTSize)
    {
        auto subcase_name = std::format("FFT size: {}", fft_size);
        SUBCASE(subcase_name.c_str())
        {
            sfFDN::FFT fft(fft_size);

            auto input_buffer = fft.AllocateRealBuffer();
            auto output_buffer = fft.AllocateComplexBuffer();

            // Fill with white noise
            std::vector<float> expected_buffer(input_buffer.size(), 0.f);
            std::default_random_engine generator;
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < input_buffer.size(); ++i)
            {
                input_buffer[i] = dist(generator);
                expected_buffer[i] = input_buffer[i];
            }

            std::fill(output_buffer.begin(), output_buffer.end(), 0.f);

            fft.Forward(input_buffer, output_buffer);
            fft.Inverse(output_buffer, input_buffer);

            const float scale = 1.0f / static_cast<float>(fft_size);
            for (size_t i = 0; i < input_buffer.size(); ++i)
            {
                CHECK(input_buffer[i] * scale == doctest::Approx(expected_buffer[i]));
            }
        }
    }
}