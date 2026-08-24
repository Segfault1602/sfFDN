#include "production_workloads.h"

#include "sffdn/sffdn.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace
{
constexpr float kSampleRate = 48000.f;

std::unique_ptr<sfFDN::FDN> CreateProductionFDN(
    uint32_t internal_block_size, std::span<const float> delays, sfFDN::ScalarMatrixType matrix_type,
    const sfFDN::attenuation_filter_variant_t& attenuation)
{
    const auto order = static_cast<uint32_t>(delays.size());
    auto fdn = std::make_unique<sfFDN::FDN>(order, internal_block_size, false);
    if (!fdn->SetInputGains(std::vector<float>(order, 0.5f)) ||
        !fdn->SetOutputGains(std::vector<float>(order, 0.5f)))
    {
        throw std::runtime_error("Failed to configure production FDN gains");
    }
    fdn->SetDirectGain(0.f);
    if (!fdn->SetDelays(delays))
    {
        throw std::runtime_error("Failed to configure production FDN delays");
    }
    if (!fdn->SetFeedbackMatrix(std::make_unique<sfFDN::ScalarFeedbackMatrix>(
            sfFDN::ScalarFeedbackMatrixOptions{.matrix_size = order, .type = matrix_type})))
    {
        throw std::runtime_error("Failed to configure production FDN feedback matrix");
    }
    if (!fdn->SetLoopFilter(sfFDN::CreateAttenuationFilterBank(attenuation, delays)))
    {
        throw std::runtime_error("Failed to configure production FDN loop filter");
    }
    return fdn;
}
} // namespace

std::vector<ProductionFDNWorkload> CreateProductionFDNWorkloads()
{
    constexpr std::array<float, 8> kSandboxDelays = {809.f, 877.f, 937.f, 1049.f,
                                                     1151.f, 1249.f, 1373.f, 1499.f};
    constexpr std::array<float, 4> kOptDelays4 = {1499.f, 1889.f, 2381.f, 2999.f};
    constexpr std::array<float, 6> kOptDelays6 = {997.f, 1153.f, 1327.f, 1559.f, 1801.f, 2099.f};
    constexpr std::array<float, 8> kOptDelays8 = {809.f, 877.f, 937.f, 1049.f,
                                                 1151.f, 1249.f, 1373.f, 1499.f};
    const auto opt_delays16 =
        sfFDN::GetDelayLengths(16, 512, 3000, sfFDN::DelayLengthType::Uniform);
    const auto opt_delays32 =
        sfFDN::GetDelayLengths(32, 512, 3000, sfFDN::DelayLengthType::Uniform);
    const sfFDN::ThreeBandFilterOptions three_band{
        .t60s = {1.5f, 1.f, 0.5f}, .delay = 0.f, .sample_rate = kSampleRate};
    const sfFDN::TenBandFilterOptions ten_band{
        .t60s = {2.f, 2.f, 1.8f, 1.6f, 1.4f, 1.2f, 1.f, 0.8f, 0.6f, 0.5f},
        .delay = 0.f,
        .sample_rate = kSampleRate,
        .shelf_cutoff = 8000.f,
    };

    std::vector<ProductionFDNWorkload> workloads;
    workloads.push_back({
        .name = "FdnSandbox N8 callback=1024 internal=64 homogeneous Hadamard",
        .callback_size = 1024,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(
            64, kSandboxDelays, sfFDN::ScalarMatrixType::Hadamard,
            sfFDN::HomogenousFilterOptions{.t60 = 1.f, .delay = 0.f, .sample_rate = kSampleRate}),
    });
    workloads.push_back({
        .name = "FdnSandbox N8 callback=1024 internal=64 homogeneous Random",
        .callback_size = 1024,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(
            64, kSandboxDelays, sfFDN::ScalarMatrixType::Random,
            sfFDN::HomogenousFilterOptions{.t60 = 1.f, .delay = 0.f, .sample_rate = kSampleRate}),
    });
    workloads.push_back({
        .name = "fdn_opt N4 block=128 three-band Hadamard",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, kOptDelays4, sfFDN::ScalarMatrixType::Hadamard, three_band),
    });
    workloads.push_back({
        .name = "fdn_opt N6 block=128 ten-band Householder",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, kOptDelays6, sfFDN::ScalarMatrixType::Householder, ten_band),
    });
    workloads.push_back({
        .name = "fdn_opt N8 block=128 ten-band Hadamard",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, kOptDelays8, sfFDN::ScalarMatrixType::Hadamard, ten_band),
    });
    workloads.push_back({
        .name = "fdn_opt N8 block=128 ten-band Random",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, kOptDelays8, sfFDN::ScalarMatrixType::Random, ten_band),
    });
    workloads.push_back({
        .name = "FDN N8 block=128 three-band Hadamard",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, kOptDelays8, sfFDN::ScalarMatrixType::Hadamard, three_band),
    });
    workloads.push_back({
        .name = "FDN N8 block=128 three-band Random",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, kOptDelays8, sfFDN::ScalarMatrixType::Random, three_band),
    });
    workloads.push_back({
        .name = "FDN N16 block=128 ten-band Hadamard",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, opt_delays16, sfFDN::ScalarMatrixType::Hadamard, ten_band),
    });
    workloads.push_back({
        .name = "FDN N16 block=128 ten-band Random",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, opt_delays16, sfFDN::ScalarMatrixType::Random, ten_band),
    });
    workloads.push_back({
        .name = "FDN N16 block=128 three-band Hadamard",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, opt_delays16, sfFDN::ScalarMatrixType::Hadamard, three_band),
    });
    workloads.push_back({
        .name = "FDN N16 block=128 three-band Random",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, opt_delays16, sfFDN::ScalarMatrixType::Random, three_band),
    });
    workloads.push_back({
        .name = "FDN N32 block=128 ten-band Hadamard",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, opt_delays32, sfFDN::ScalarMatrixType::Hadamard, ten_band),
    });
    workloads.push_back({
        .name = "FDN N32 block=128 ten-band Random",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, opt_delays32, sfFDN::ScalarMatrixType::Random, ten_band),
    });
    workloads.push_back({
        .name = "FDN N32 block=128 three-band Hadamard",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, opt_delays32, sfFDN::ScalarMatrixType::Hadamard, three_band),
    });
    workloads.push_back({
        .name = "FDN N32 block=128 three-band Random",
        .callback_size = 128,
        .sample_rate = kSampleRate,
        .fdn = CreateProductionFDN(128, opt_delays32, sfFDN::ScalarMatrixType::Random, three_band),
    });
    return workloads;
}
