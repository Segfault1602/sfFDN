# Multi-Channel Processors

This section describes the multi-channel processors provided by sfFDN. These are audio processors that operate on multiple channels of audio simultaneously. They can be used at the end of the input gains block, in the beginning of the output gains block and in the feedback paths of the delay network.

## Processors

- [Gains](@ref sfFDN::ParallelGains): A simple processor that applies a gain to each channel independently. Must be used in the [ParallelGainsMode::Parallel](@ref sfFDN::ParallelGainsMode) mode.
- [Schroeder Allpass](@ref sfFDN::MultichannelSchroederAllpassSectionOptions): A parallel bank of [Schroeder allpass filters](@ref sfFDN::SchroederAllpassSection).
- [Time-Varying Schroeder Allpass](@ref sfFDN::MultichannelTimeVaryingSchroederAllpassSectionOptions): A parallel bank of energy-preserving [time-varying Schroeder allpass sections](@ref sfFDN::TimeVaryingSchroederAllpassSection). Each stage modulates its gain while retaining a fixed integer delay. Series sections are suitable for lossless FDN feedback paths when every configured gain range remains strictly inside `(-1, 1)`; a parallel section sums its stage outputs and is not generally lossless when it contains more than one stage.
- [Dattorro Delay Bank](@ref sfFDN::MultichannelDattorroDelayOptions): A parallel bank of [Dattorro delay-line effects](@ref sfFDN::DattorroDelay), one per channel. See [MakeMultichannelDattorroDelayOptions](@ref sfFDN::MakeMultichannelDattorroDelayOptions) for a decorrelated preset, which staggers the modulation of each channel and uses allpass interpolation so that the magnitude response stays flat. Note that only the presets without feedback are safe to place in the feedback path: a modulated [white chorus](@ref sfFDN::DattorroEffectType) or flanger has a peak gain of roughly +15 dB and will make the network diverge. See [MakeMultichannelDattorroDelayOptions](@ref sfFDN::MakeMultichannelDattorroDelayOptions) for the per-preset figures.
- [Delay bank](@ref sfFDN::DelayBank): A parallel bank of delay lines. Each delay line can have a different length and can be configured to use interpolation for fractional delay lengths.
- [Time-varying Delay Bank](@ref sfFDN::DelayBankTimeVarying): A parallel bank of time-varying delay lines. The delay lengths are modulated over time using a sine wave.
- [Feedback Matrix](@ref sfFDN::ScalarFeedbackMatrix): Simple feedback matrix with scalar coefficients.
- [Filter Feedback Matrix](@ref sfFDN::FilterFeedbackMatrix): Implementation of a Filter Feedback Matrix based on the design by S. J. Schlecht and E. A. P. Habets, “Scattering in feedback delay networks.” A filter feedback matrix consists of a series of scalar matrix interleaved with banks of delay lines.
- [Attenuation Filter Bank](@ref sfFDN::AttenuationFilterBankOptions): A parallel bank of attenuation filters. These filters are usually designed to target a specific RT60 and their gains are scaled according to the length of the delay lines. See also the [Filtering](filters.md) manual page for the four attenuation filter variants and the associated design helpers.


## AudioProcessorChain

The [AudioProcessorChain](@ref sfFDN::AudioProcessorChain) class allows you to chain multiple multi-channel processors together. This is useful for creating more complex processing chains without having to create a custom processor class. You can add any of the multi-channel processors to the chain, as long as they have the same number of channels, and they will be processed in the order they were added.