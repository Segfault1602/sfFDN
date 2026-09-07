// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "processor_factory.h"

#include "processor_option_validation.h"
#include "sffdn/dattorro_delay.h"
#include "sffdn/delay.h"
#include "sffdn/delay_time_varying.h"
#include "sffdn/filter.h"
#include "sffdn/filter_design.h"
#include "sffdn/nonlinear.h"
#include "sffdn/schroeder_allpass.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace sfFDN
{
std::unique_ptr<AudioProcessor> CreateSingleChannelProcessor(const single_channel_processor_variant_t& config)
{
    return std::visit<std::unique_ptr<AudioProcessor>>(
        overloaded{[](const SchroederAllpassSectionOptions& options) {
                       return std::make_unique<SchroederAllpassSection>(options);
                   },
                   [](const TimeVaryingSchroederAllpassSectionOptions& options) {
                       try
                       {
                           return std::make_unique<TimeVaryingSchroederAllpassSection>(options);
                       }
                       catch (const std::invalid_argument& error)
                       {
                           throw std::runtime_error(
                               std::string("Invalid time-varying Schroeder allpass configuration: ") + error.what());
                       }
                   },
                   [](const AllpassFilterOptions& options) {
                       detail::RequireValidOptions(options);
                       return std::make_unique<AllpassFilter>(options);
                   },
                   [](const CascadedBiquadsOptions& options) {
                       detail::RequireValidOptions(options);
                       return std::make_unique<CascadedBiquads>(options);
                   },
                   [](const FirOptions& options) { return MakeFirFilter(options); },
                   [](const DelayOptions& options) -> std::unique_ptr<AudioProcessor> {
                       if (options.lfo_config.has_value())
                       {
                           return std::make_unique<DelayTimeVarying>(options);
                       }
                       return std::make_unique<DelayInterp>(options);
                   },
                   [](const GraphicEQOptions& options) {
                       detail::RequireValidOptions(options);
                       const auto coefficients = DesignGraphicEQ(options);
                       const CascadedBiquadsOptions filter_options{
                           std::vector<FilterCoefficients>(coefficients.begin(), coefficients.end()),
                       };
                       return std::make_unique<CascadedBiquads>(filter_options);
                   },
                   [](const DattorroDelayOptions& options) { return std::make_unique<DattorroDelay>(options); },
                   [](const ControllableFullWaveRectifierOptions& options) {
                       detail::RequireValidOptions(options);
                       return std::make_unique<ControllableFullWaveRectifier>(options);
                   },
                   [](const SignalDependentFractionalDelayOptions& options) {
                       detail::RequireValidOptions(options);
                       return std::make_unique<SignalDependentFractionalDelay>(options);
                   },
                   [](const RingModulatorOptions& options) {
                       detail::RequireValidOptions(options);
                       return std::make_unique<RingModulator>(options);
                   },},
        config);
}
} // namespace sfFDN
