# Filtering

This section documents the attenuation filters used by sfFDN to control decay time inside an FDN loop. The public configuration entry point is the variant type [sfFDN::attenuation_filter_variant_t](@ref sfFDN::attenuation_filter_variant_t), which can hold one of four filter configurations:

- [sfFDN::HomogenousFilterOptions](@ref sfFDN::HomogenousFilterOptions)
- [sfFDN::TwoBandFilterOptions](@ref sfFDN::TwoBandFilterOptions)
- [sfFDN::ThreeBandFilterOptions](@ref sfFDN::ThreeBandFilterOptions)
- [sfFDN::TenBandFilterOptions](@ref sfFDN::TenBandFilterOptions)

Use [sfFDN::FilterDesigner](@ref sfFDN::FilterDesigner), initialized with the intended sampling
rate, to design coefficients or to create attenuation processors. The designer is setup-time state:
created processors retain their coefficients and do not retain a reference to it.

[sfFDN::CreateAttenuationFilter](@ref sfFDN::CreateAttenuationFilter) turns one variant into a
concrete single-channel processor, and [sfFDN::CreateAttenuationFilterBank](@ref
sfFDN::CreateAttenuationFilterBank) builds a parallel bank. Both require the explicit designer.

## Filter Variants

### Homogenous Filter

[sfFDN::HomogenousFilterOptions](@ref sfFDN::HomogenousFilterOptions) describes the simplest attenuation filter. It applies the same decay characteristic across all frequencies by converting a target T60 value into a frequency-independent feedback gain. The gain is calculated using the formula:

\begin{equation}
  g_i = 10^{\frac{-3m_i}{T_{60}f_s}},
\end{equation}

where \f$ m_i \f$ is the delay length of the \f$ i \f$-th delay line in samples, \f$ T_{60} \f$ is the target T60 in seconds, and \f$ f_s \f$ is the sampling rate in Hz.

Use this option when you want a compact, low-cost decay control and do not need frequency-dependent decay shaping.

Key fields:

- `t60`: Target T60 in seconds.
- `delay`: Delay in samples used to derive the decay gain. A non-positive value is derived from
  the primary delay bank when the filter is placed in an `FDNConfig` loop or dedicated attenuation slot.

### Two-Band Filter

[sfFDN::TwoBandFilterOptions](@ref sfFDN::TwoBandFilterOptions) configures a two-band attenuation filter with independent decay targets at DC and Nyquist.

The resulting filter is designed by `FilterDesigner::DesignFilter`, which produces a one-pole
absorption filter as proposed by Jot and Chaigne in [1]. The design procedure uses the two T60
targets to derive the filter coefficient.

Key fields:

- `t60s`: Two target T60 values, for low and high frequencies.
- `delay`: Delay in samples used by the design equation. A non-positive value is derived from the
  primary delay bank when the filter is placed in an `FDNConfig` loop or dedicated attenuation slot.

### Three-Band Filter

[sfFDN::ThreeBandFilterOptions](@ref sfFDN::ThreeBandFilterOptions) adds a middle band to the attenuation design. It uses two shelf frequencies and three T60 targets to shape decay over low, mid, and high frequency ranges. The resulting filter is composed of a 2nd-order low-shelf filter and a 2nd-order high-shelf filter.

The filter is designed by `FilterDesigner::DesignFilter`, which returns a cascade of biquad
sections suitable for direct use in the loop filter path.

Key fields:

- `t60s`: Three target T60 values for low, mid, and high bands.
- `delay`: Delay in samples used by the design equation. A non-positive value is derived from the
  primary delay bank when the filter is placed in an `FDNConfig` loop or dedicated attenuation slot.
- `freqs`: Shelf crossover frequencies that separate the three bands.
- `q`: Shelf Q factor used in the filter design.

### Ten-Band Filter

[sfFDN::TenBandFilterOptions](@ref sfFDN::TenBandFilterOptions) provides the most detailed attenuation control in the public API. It targets ten octave-style bands and is implemented as a cascade of second-order biquad sections.

The filter is designed by `FilterDesigner::DesignFilter`, which follows the two-stage attenuation
filter method described in [2].

Key fields:

- `t60s`: Ten target T60 values, one per band.
- `delay`: Delay in samples used by the design equation. A non-positive value is derived from the
  primary delay bank when the filter is placed in an `FDNConfig` loop or dedicated attenuation slot.
- `shelf_cutoff`: Shelf crossover used by the design procedure.

## Design Helpers

`FilterDesigner::DesignFilter` has typed overloads for homogeneous (a linear gain), two-band (a
`std::pair<float, float>`), three-band (two `FilterCoefficients`), ten-band (eleven
`FilterCoefficients`), and Graphic EQ (eleven `FilterCoefficients`) designs. It also designs RBJ
low shelves, high shelves, and cookbook peaking EQs from options containing `frequency` in Hz,
`gain_db` in dB, and positive dimensionless `q`.

`FilterDesigner::T60ToGain(t60_seconds, delay_samples)` returns the linear amplitude gain
`10^(-3 * delay_samples / (t60_seconds * sample_rate))`. T60 is in seconds and delay is in
samples; both arguments are explicit.

```c++
const sfFDN::FilterDesigner designer(48000.0f);
const auto one_pole = designer.DesignFilter(
    sfFDN::TwoBandFilterOptions{.t60s = {1.5f, 0.7f}, .delay = 1007.0f});
const auto peaking = designer.DesignFilter(
    sfFDN::PeakingOptions{.frequency = 2000.0f, .gain_db = -3.0f, .q = 1.0f});
const float gain = designer.T60ToGain(1.5f, 1007.0f);
```

RBJ designs return one `FilterCoefficients` section with `a0 == 1`. Shelf `q` is the quality factor. Frequencies must be strictly between zero and
Nyquist, and Q must be positive; invalid values are rejected rather than clamped. Changing the
rate requires another designer and does not reconfigure existing processors.

To create a single filter from one of the variants, construct a designer and call
`CreateAttenuationFilter`. The `delay` field must be set to a valid value.

```c++
sfFDN::TwoBandFilterOptions options;
options.t60s = { 1.0f, 0.5f };
options.delay = 1007;

const sfFDN::FilterDesigner designer(48000.0f);
auto filter = sfFDN::CreateAttenuationFilter(options, designer);
```

To create a parallel bank of filters, use sfFDN::CreateAttenuationFilterBank.

```c++
sfFDN::AttenuationFilterBankOptions options;

sfFDN::TwoBandFilterOptions options1;
options1.t60s = {1.0f, 0.5f};
options1.delay = 1007;
options.filter_configs.emplace_back(options1);

sfFDN::TwoBandFilterOptions options2;
options2.t60s = {1.0f, 0.5f};
options2.delay = 1433;
options.filter_configs.emplace_back(options2);

const sfFDN::FilterDesigner designer(48000.0f);
auto filter_bank = sfFDN::CreateAttenuationFilterBank(options, designer);
```

A version accepting one `sfFDN::attenuation_filter_variant_t`, a list of delay lengths, and a
designer is also provided. This is useful when the same filter design is desired across all delay
lines, with only the delay length differing.

For a `MultichannelProcessorOptions` bank containing `GraphicEQOptions`, use
`FilterBank(options, designer)`. The empty `FilterBank` constructor remains available for
manually assembled processor banks.

An FDN loop or dedicated attenuation slot accepts either one shared attenuation configuration or
one configuration per delay line. For an FDN with more than one channel, a shared entry is expanded
and its delay is derived from each primary delay, even if the entry supplied a positive delay.
An exactly sized bank retains positive explicit delays and derives nonpositive delays from the
corresponding primary delays. For order one, a one-entry bank is already exactly sized.

Input and output multichannel insertion slots require one configuration per channel with
nonnegative explicit delays; these placements do not infer delays from the primary loop. In an
`FDNConfig`, the root `FDNConfig::sample_rate` supplies the rate for attenuation and Graphic EQ
designs. Nested filter `sample_rate` fields are ignored if present in JSON, including when they
disagree with the root rate. This does not override unrelated processors such as the nonlinear
rectifier, which retain their own rate setting.

## References

[1]Jot, J.-M., & Chaigne, A. (1991). Digital delay networks for designing artificial reverberators. Proceedings of the 90th Audio Engineering Society Convention (AES), (3030), 1–16.

[2] Välimäki, V., Prawda, K., & Schlecht, S. J. (2024). Two-Stage Attenuation Filter for Artificial Reverberation. IEEE Signal Processing Letters, 31, 391–395. https://doi.org/10.1109/LSP.2024.3352510
