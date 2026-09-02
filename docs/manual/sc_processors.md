# Single-Channel Processors

This section describes the single-channel processors provided by sfFDN. These are audio processors that operate on a single channel of audio. They can be used at the beginning of the input gains block, at the end of the output gains block, or they can be combined in a [FilterBank](@ref sfFDN::FilterBank) object and used as a multi-channel processor.

## Processors

- [Delay](@ref sfFDN::DelayInterp): A simple delay line that supports interpolation for fractional delay lengths.
- [Dattorro Delay](@ref sfFDN::DattorroDelay): A modulated delay line with blend, feedforward and feedback controls, as described by Jon Dattorro in "Effect Design Part 2: Delay-Line Modulation and Chorus". Vibrato, flanging, chorus, doubling and echo are all obtained from this one processor; see [MakeDattorroDelayOptions](@ref sfFDN::MakeDattorroDelayOptions).
- [Schroeder Allpass](@ref sfFDN::SchroederAllpass): An allpass filter consisting of a delay line and a feedback and feedforward path. See also [SchroederAllpassSection](@ref sfFDN::SchroederAllpassSection), which implements a group of Schroeder allpass filters placed in series or in parallel.
- [Time-Varying Schroeder Allpass](@ref sfFDN::TimeVaryingSchroederAllpass): An energy-preserving allpass whose gain coefficient can be modulated while its integer delay remains fixed. [TimeVaryingSchroederAllpassSection](@ref sfFDN::TimeVaryingSchroederAllpassSection) groups stages in series or parallel.
- [FIR](@ref sfFDN::Fir): A finite impulse response filter.
- [Sparse FIR](@ref sfFDN::SparseFir): A sparse finite impulse response filter that allows for non-uniformly spaced coefficients.
- [IIR](@ref sfFDN::CascadedBiquads): An infinite impulse response filter implemented as a cascade of biquad sections.
- [One Pole](@ref sfFDN::OnePoleFilter): A simple one-pole filter.
- [Allpass](@ref sfFDN::Allpass): A first-order allpass filter.
- [Controllable Full-Wave Rectifier](@ref sfFDN::ControllableFullWaveRectifier): A waveshaper that blends between the input and its full-wave rectification, with first-order antiderivative antialiasing and an optional dc blocker. Generates even harmonics.
- [Signal-Dependent Fractional Delay](@ref sfFDN::SignalDependentFractionalDelay): Delays the positive and negative half-wave components of the input by different fractional amounts, distorting the waveform around its zero crossings. A milder alternative to the rectifier.
- [Ring Modulator](@ref sfFDN::RingModulator): Multiplies the input by a sinusoid, replacing its spectrum with two sidebands.

## AudioProcessorChain

The [AudioProcessorChain](@ref sfFDN::AudioProcessorChain) class allows you to chain multiple single-channel processors together. This is useful for creating more complex processing chains without having to create a custom processor class. You can add any of the single-channel processors to the chain, and they will be processed in the order they were added.

## Energy-Preserving Time-Varying Allpass

`TimeVaryingSchroederAllpass` implements the normalized Type V structure from J. Werner, "Energy-Preserving Time-Varying Schroeder Allpass Filters," DAFx-20. For input `x`, delayed state `w`, instantaneous gain `g`, and `c = sqrt(1 - g*g)`, it computes `y = c*w - g*x` and stores `u = c*x + g*w`. This orthogonal transform preserves `y*y + u*u = x*x + w*w` at every sample.

Here, "time-varying" means gain modulation only: the delay is a fixed integer number of samples. Fractional or modulated delay lengths are outside the energy-preservation guarantee. `ModulationOptions::frequency` is measured in cycles per sample, `amplitude` is the peak gain deviation, and both must be non-zero. The complete range must satisfy `abs(base_gain) + abs(amplitude) < 1`. Use [SchroederAllpass](@ref sfFDN::SchroederAllpass) for a fixed-gain allpass.