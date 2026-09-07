# Multi-Channel Processors

This section describes the multi-channel processors provided by sfFDN. These are audio processors that operate on multiple channels of audio simultaneously. They can be used at the end of the input gains block, in the beginning of the output gains block and in the feedback paths of the delay network.

## Processors

- [Channel matrix](@ref sfFDN::ChannelMatrix): Applies a static dense rectangular matrix between input and output
  channels. Coefficients are row-major by output row and input column. Processing overwrites the destination, requires
  disjoint input/output storage, and performs no allocation.
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
  arrangement, not an instruction to apply \f$A^T\f$. Its source is either a generated recipe or explicit,
  shape-checked row-major `MatrixData`, which owns its values and exposes `Values()` spans.
- [Filter Feedback Matrix](@ref sfFDN::FilterFeedbackMatrix): Implementation of a Filter Feedback Matrix based on the design by S. J. Schlecht and E. A. P. Habets, “Scattering in feedback delay networks.” A filter feedback matrix consists of a series of scalar matrix interleaved with banks of delay lines.
  `CascadedFeedbackMatrixOptions::rng_seed` controls both the generated matrices and delay lengths.
- [Attenuation Filter Bank](@ref sfFDN::AttenuationFilterBankOptions): A parallel bank of attenuation filters. These filters are usually designed to target a specific RT60 and their gains are scaled according to the length of the delay lines. See also the [Filtering](filters.md) manual page for the four attenuation filter variants and the associated design helpers.


## AudioProcessorChain

The [AudioProcessorChain](@ref sfFDN::AudioProcessorChain) class allows you to chain multiple multi-channel processors together. This is useful for creating more complex processing chains without having to create a custom processor class. You can add any of the multi-channel processors to the chain, as long as they have the same number of channels, and they will be processed in the order they were added.

## Configuration

`MultichannelProcessorOptions` configures a `FilterBank` with one single-channel processor per entry. An entry may
hold any `single_channel_processor_variant_t`; `std::nullopt` is a pass-through channel.

```json
{"MultichannelProcessorOptions":{"channels":[{"FirOptions":{"coeffs":[1.0]}},null]}}
```

For example, construct a two-channel FIR bank:

```cpp
MultichannelProcessorOptions fir_bank{
    .channels = {FirOptions{.coeffs = {1.f}}, FirOptions{.coeffs = {0.5f, 0.5f}}},
};
FilterBank processor(fir_bank);
```

Preset helpers also return `MultichannelProcessorOptions`:

```cpp
const auto dattorro = MakeMultichannelDattorroDelayOptions(DattorroEffectType::Vibrato, 48000.f, 8);
FilterBank processor(dattorro);
```

For an FDN placement, the bank contains one entry per FDN channel. `ParallelGainsOptions` retains its explicit
`ParallelGainsMode` for standalone and multichannel processors. `StageGainsOptions` configures FDN input and output
stages: an unmodulated gain list is constructed as a rectangular `ChannelMatrix`, while a nonempty modulation vector
uses `TimeVaryingParallelGains`.

### Matrix sources and seeds

Scalar feedback matrices have one `source`: a generated `GeneratedMatrixOptions` recipe or explicit
`MatrixData`. `MatrixData(order, coefficients)` owns a row-major vector and rejects a coefficient count other than
`order * order`; `Values()` returns mutable or const spans over that data.

```c++
// Generated matrix
ScalarFeedbackMatrixOptions generated{
    .source = GeneratedMatrixOptions{
        .matrix_size = order,
        .generator = ScalarMatrixType::Hadamard,
        .rng_seed = kDefaultMatrixSeed}};

// Explicit data
ScalarFeedbackMatrixOptions explicit_matrix{
    .source = MatrixData{2, {1.f, 0.f, 0.f, 1.f}}};

// Parameterized generation
ScalarFeedbackMatrixOptions diffusion{
    .source = GeneratedMatrixOptions{
        .matrix_size = order,
        .generator = VariableDiffusionOptions{.diffusion = 0.5f},
        .rng_seed = 0U}};
```

Matrix recipes default to `kDefaultMatrixSeed`. Zero is also a deterministic seed, not a request for randomization.

Use `RandomizeMatrixSeeds(config)` to assign new random seeds to generated matrices in the feedback, input, output,
and loop stages. Explicit `MatrixData` is unchanged.
