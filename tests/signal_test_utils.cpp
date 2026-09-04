#include "signal_test_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "filter_coeffs.h"
#include "test_utils.h"

namespace sfFDNTest
{
SignalComparison CompareSignals(std::span<const float> reference, std::span<const float> actual)
{
    SignalComparison result;
    if (reference.size() != actual.size())
    {
        result.max_absolute_error = std::numeric_limits<float>::infinity();
        result.error_energy = std::numeric_limits<double>::infinity();
        result.snr_db = -std::numeric_limits<double>::infinity();
        return result;
    }

    for (size_t i = 0; i < reference.size(); ++i)
    {
        const double error = static_cast<double>(actual[i]) - reference[i];
        result.max_absolute_error = std::max(result.max_absolute_error, static_cast<float>(std::abs(error)));
        result.signal_energy += static_cast<double>(reference[i]) * reference[i];
        result.error_energy += error * error;
    }

    result.exact_match = result.error_energy == 0.0;
    if (result.exact_match)
    {
        result.snr_db = std::numeric_limits<double>::infinity();
    }
    else if (result.signal_energy == 0.0)
    {
        result.snr_db = -std::numeric_limits<double>::infinity();
    }
    else
    {
        result.snr_db = 10.0 * std::log10(result.signal_energy / result.error_energy);
    }
    return result;
}

void RequireSignalsClose(std::span<const float> reference, std::span<const float> actual, float max_absolute_error,
                         double minimum_snr_db)
{
    const auto comparison = CompareSignals(reference, actual);
    INFO("max absolute error: " << comparison.max_absolute_error << ", signal energy: " << comparison.signal_energy
                                << ", error energy: " << comparison.error_energy << ", SNR: " << comparison.snr_db
                                << " dB");
    REQUIRE(reference.size() == actual.size());
    REQUIRE(comparison.max_absolute_error <= max_absolute_error);
    REQUIRE(comparison.snr_db >= minimum_snr_db);
}

std::vector<float> RenderMonoBlocks(std::span<const float> input, uint32_t block_size, uint32_t tail_samples,
                                    const MonoBlockProcessor& process_block)
{
    if (block_size == 0)
    {
        throw std::invalid_argument("block_size must be nonzero");
    }

    const size_t sample_count = input.size() + tail_samples;
    const size_t rendered_sample_count = ((sample_count + block_size - 1) / block_size) * block_size;
    std::vector<float> padded_input(rendered_sample_count, 0.f);
    std::ranges::copy(input, padded_input.begin());
    std::vector<float> output(rendered_sample_count, 0.f);

    for (size_t start = 0; start < rendered_sample_count; start += block_size)
    {
        process_block(std::span(padded_input).subspan(start, block_size), std::span(output).subspan(start, block_size));
    }

    output.resize(sample_count);
    return output;
}

std::unique_ptr<sfFDN::CascadedBiquads> CreateReferenceAbsorptionFilter()
{
    auto filter = std::make_unique<sfFDN::CascadedBiquads>();
    filter->SetCoefficients(k_h001_AbsorbtionSOS[0]);
    return filter;
}

std::vector<float> CreateReferenceAbsorptionFir()
{
    auto filter = CreateReferenceAbsorptionFilter();
    return GetImpulseResponse(filter.get());
}
} // namespace sfFDNTest
