# Multi-Channel Processors

This section describes the multi-channel processors provided by sfFDN. These are audio processors that operate on multiple channels of audio simultaneously. They can be used at the end of the input gains block, in the beginning of the output gains block and in the feedback paths of the delay network.

## Processors

- [Gains](@ref sfFDN::ParallelGains): A simple processor that applies a gain to each channel independently. Must be used in the [ParallelGainsMode::Parallel](@ref sfFDN::ParallelGainsMode) mode.
- [Generic processor bank](@ref sfFDN::MultichannelProcessorOptions): A [FilterBank](@ref sfFDN::FilterBank)
  with one independently constructed single-channel processor per channel. Entries may mix any
  `single_channel_processor_variant_t` type; `std::nullopt` is a pass-through channel. The bank's channel count is
  exactly `channels.size()`, and it must equal `FDNConfig::fdn_size` in every FDN placement. For example:
  `MultichannelProcessorOptions{.channels = {FirOptions{.coeffs = {1.f}}, std::nullopt,
  SignalDependentFractionalDelayOptions{.d = 0.5f}}}`. Construct the same bank directly with
  `FilterBank(options)`.
- [Time-varying Schroeder allpass](@ref sfFDN::TimeVaryingSchroederAllpassSection): add one option per generic-bank
  channel. Series sections remain suitable for lossless feedback paths while every gain range stays strictly inside
  `(-1, 1)`; a parallel section with multiple stages is not generally lossless.
- [Dattorro Delay](@ref sfFDN::DattorroDelay): use
  [MakeMultichannelDattorroDelayOptions](@ref sfFDN::MakeMultichannelDattorroDelayOptions) for a decorrelated
  generic bank. It staggers modulation and uses allpass interpolation. Only presets without feedback are safe in an
  FDN feedback path: a modulated [white chorus](@ref sfFDN::DattorroEffectType) or flanger can peak around +15 dB.
- [Delay bank](@ref sfFDN::DelayBank): A parallel bank of delay lines. Each delay line can have a different length and can be configured to use interpolation for fractional delay lengths.
- [Time-varying Delay Bank](@ref sfFDN::DelayBankTimeVarying): A parallel bank of time-varying delay lines. The delay lengths are modulated over time using a sine wave.
- [Feedback Matrix](@ref sfFDN::ScalarFeedbackMatrix): Simple feedback matrix with scalar coefficients. Public
  coefficients are row-major (`matrix[row * N + column] = A[row, column]`) and apply \f$y = A x\f$; this is the
  same convention as a pyFDN/NumPy `(out, in)` matrix flattened with
  `numpy.asarray(A).ravel(order="C")` and evaluated as `x @ A.T`. The FDN's transposed topology is a signal-flow
  arrangement, not an instruction to apply \f$A^T\f$. Set a nonzero `rng_seed` to reproduce a generated random
  gallery matrix; zero retains nondeterministic random generation. `custom_matrix`, when supplied, takes precedence
  over matrix generation.
- [Filter Feedback Matrix](@ref sfFDN::FilterFeedbackMatrix): Implementation of a Filter Feedback Matrix based on the design by S. J. Schlecht and E. A. P. Habets, “Scattering in feedback delay networks.” A filter feedback matrix consists of a series of scalar matrix interleaved with banks of delay lines.
  A nonzero `CascadedFeedbackMatrixOptions::rng_seed` reproduces every generated stage matrix and stage delay
  distribution; zero retains nondeterministic random generation.
- [Attenuation Filter Bank](@ref sfFDN::AttenuationFilterBankOptions): A parallel bank of attenuation filters. These filters are usually designed to target a specific RT60 and their gains are scaled according to the length of the delay lines. See also the [Filtering](filters.md) manual page for the four attenuation filter variants and the associated design helpers.


## AudioProcessorChain

The [AudioProcessorChain](@ref sfFDN::AudioProcessorChain) class allows you to chain multiple multi-channel processors together. This is useful for creating more complex processing chains without having to create a custom processor class. You can add any of the multi-channel processors to the chain, as long as they have the same number of channels, and they will be processed in the order they were added.

## Migration and JSON

The former homogeneous independent-channel option wrappers are removed. Rebuild C++ callers with
`MultichannelProcessorOptions`; saved configurations must be migrated and are not read automatically. The canonical
JSON form is:

```json
{"MultichannelProcessorOptions":{"channels":[{"FirOptions":{"coeffs":[1.0]}},null]}}
```

| Former wrapper field | Generic-bank replacement |
| --- | --- |
| Schroeder allpass `sections` | One `SchroederAllpassSectionOptions` per `channels` entry |
| Time-varying Schroeder allpass `sections` | One `TimeVaryingSchroederAllpassSectionOptions` per `channels` entry |
| Dattorro `delays` | One `DattorroDelayOptions` per `channels` entry |
| FIR `coeffs` | One `FirOptions{.coeffs = coefficients}` per coefficient vector |
| Nonlinear `channels` | The same ordered entries; replace an old bypass with `std::nullopt` |

The old `MakeMultichannel...` builder functions are replaced by `FilterBank(options)`. Useful preset functions,
including `MakeMultichannelDattorroDelayOptions()` and the nonlinear preset helpers, remain and now return
`MultichannelProcessorOptions`. An all-null bank is a valid identity bank, and an empty standalone `FilterBank` is
valid; an FDN placement still requires exactly its positive `fdn_size` channel count.

For example, replace a homogeneous FIR wrapper with a generic bank:

```cpp
// Before: MultichannelFirOptions{.coeffs = {{1.f}, {0.5f, 0.5f}}}
MultichannelProcessorOptions fir_bank{
    .channels = {FirOptions{.coeffs = {1.f}}, FirOptions{.coeffs = {0.5f, 0.5f}}},
};
FilterBank processor(fir_bank);
```

The same representation handles nonlinear bypass and preset banks without special factories:

```cpp
auto shimmer = MakeMultichannelControllableFullWaveRectifierOptions(1.f, 48000.f, 8, 2);
// shimmer.channels[0..5] are nullopt and channels 6 and 7 hold rectifier options.
FilterBank shimmer_processor(shimmer);

auto dattorro = MakeMultichannelDattorroDelayOptions(DattorroEffectType::Vibrato, 48000.f, 8);
config.loop_filter_configs.emplace_back(dattorro);
```

Specialized attenuation filters, delay banks, gains, and feedback matrices remain distinct because they have
cross-channel or delay-dependent behavior that a generic FilterBank does not provide.

JSON readers reject unknown enum strings, malformed array shapes, and ambiguous tagged wrappers. A single-channel,
multichannel, feedback-matrix, or attenuation-filter wrapper contains exactly one supported type tag. Reads are transactional, so a failed parse does not modify an existing
options object.
