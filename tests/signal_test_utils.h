#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "sffdn/sffdn.h"

namespace sfFDNTest
{
struct SignalComparison
{
    float max_absolute_error = 0.f;
    double signal_energy = 0.0;
    double error_energy = 0.0;
    double snr_db = 0.0;
    bool exact_match = false;
};

using MonoBlockProcessor = std::function<void(std::span<float> input, std::span<float> output)>;

[[nodiscard]] SignalComparison CompareSignals(std::span<const float> reference, std::span<const float> actual);
void RequireSignalsClose(std::span<const float> reference, std::span<const float> actual, float max_absolute_error,
                         double minimum_snr_db);

[[nodiscard]] std::vector<float> RenderMonoBlocks(std::span<const float> input, uint32_t block_size,
                                                  uint32_t tail_samples, const MonoBlockProcessor& process_block);

[[nodiscard]] std::unique_ptr<sfFDN::CascadedBiquads> CreateReferenceAbsorptionFilter();
[[nodiscard]] std::vector<float> CreateReferenceAbsorptionFir();
} // namespace sfFDNTest
