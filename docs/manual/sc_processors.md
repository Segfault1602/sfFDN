# Single-Channel Processors

This section describes the single-channel processors provided by sfFDN. These are audio processors that operate on a single channel of audio. They can be used at the beginning of the input gains block, at the end of the output gains block, or they can be combined in a [FilterBank](@ref sfFDN::FilterBank) object and used as a multi-channel processor.

## Processors

- [Delay](@ref sfFDN::DelayInterp): A simple delay line that supports interpolation for fractional delay lengths.
- [Schroeder Allpass](@ref sfFDN::SchroederAllpass): An allpass filter consisting of a delay line and a feedback and feedforward path. See also [SchroederAllpassSection](@ref sfFDN::SchroederAllpassSection), which implements a group of Schroeder allpass filters placed in series or in parallel.
- [FIR](@ref sfFDN::Fir): A finite impulse response filter.
- [Sparse FIR](@ref sfFDN::SparseFir): A sparse finite impulse response filter that allows for non-uniformly spaced coefficients.
- [IIR](@ref sfFDN::CascadedBiquads): An infinite impulse response filter implemented as a cascade of biquad sections.
- [One Pole](@ref sfFDN::OnePoleFilter): A simple one-pole filter.
- [Allpass](@ref sfFDN::Allpass): A first-order allpass filter.

## AudioProcessorChain

The [AudioProcessorChain](@ref sfFDN::AudioProcessorChain) class allows you to chain multiple single-channel processors together. This is useful for creating more complex processing chains without having to create a custom processor class. You can add any of the single-channel processors to the chain, and they will be processed in the order they were added.