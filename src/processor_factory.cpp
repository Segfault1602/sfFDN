// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "processor_factory.h"

#include "sffdn/dattorro_delay.h"
#include "sffdn/delay.h"
#include "sffdn/delay_time_varying.h"
#include "sffdn/filter.h"
#include "sffdn/filter_design.h"
#include "sffdn/nonlinear.h"
#include "sffdn/schroeder_allpass.h"

#include <stdexcept>
#include <string>

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
                       auto filter = std::make_unique<AllpassFilter>();
                       filter->SetCoefficients(options.coeff);
                       return filter;
                   },
                   [](const CascadedBiquadsOptions& options) {
                       auto filter = std::make_unique<CascadedBiquads>();
                       filter->SetCoefficients(options.coeffs);
                       return filter;
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
                       auto filter = std::make_unique<CascadedBiquads>();
                       filter->SetCoefficients(DesignGraphicEQ(options));
                       return filter;
                   },
                   [](const DattorroDelayOptions& options) { return std::make_unique<DattorroDelay>(options); },
                   [](const ControllableFullWaveRectifierOptions& options) {
                       return std::make_unique<ControllableFullWaveRectifier>(options);
                   },
                   [](const SignalDependentFractionalDelayOptions& options) {
                       return std::make_unique<SignalDependentFractionalDelay>(options);
                   },
                   [](const RingModulatorOptions& options) { return std::make_unique<RingModulator>(options); },},
        config);
}
} // namespace sfFDN
