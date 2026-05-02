# Filtering

This section documents the attenuation filters used by sfFDN to control decay time inside an FDN loop. The public configuration entry point is the variant type [sfFDN::attenuation_filter_variant_t](@ref sfFDN::attenuation_filter_variant_t), which can hold one of four filter configurations:

- [sfFDN::HomogenousFilterOptions](@ref sfFDN::HomogenousFilterOptions)
- [sfFDN::TwoBandFilterOptions](@ref sfFDN::TwoBandFilterOptions)
- [sfFDN::ThreeBandFilterOptions](@ref sfFDN::ThreeBandFilterOptions)
- [sfFDN::TenBandFilterOptions](@ref sfFDN::TenBandFilterOptions)

Use [sfFDN::CreateAttenuationFilter](@ref sfFDN::CreateAttenuationFilter) to turn one variant into a concrete single-channel processor, or [sfFDN::CreateAttenuationFilterBank](@ref sfFDN::CreateAttenuationFilterBank) to build a parallel bank of attenuation filters from a bank configuration.

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
- `delay`: Delay in samples used to derive the decay gain. If this value if <= 0, sfFDN will automatically update it from the surrounding FDN configuration.
- `sample_rate`: Sampling rate used during gain calculation.

### Two-Band Filter

[sfFDN::TwoBandFilterOptions](@ref sfFDN::TwoBandFilterOptions) configures a two-band attenuation filter with independent decay targets at DC and Nyquist.

The resulting filter is designed by [sfFDN::DesignTwoBandAbsorption](@ref sfFDN::DesignTwoBandAbsorption), which produces a one-pole absorption filter as proposed by Jot and Chaigne in [1]. The design procedure uses the two T60 targets to derive the filter coefficient.

Key fields:

- `t60s`: Two target T60 values, for low and high frequencies.
- `delay`: Delay in samples used by the design equation. If this value if <= 0, sfFDN will automatically update it from the surrounding FDN configuration.
- `sample_rate`: Sampling rate used during design.

### Three-Band Filter

[sfFDN::ThreeBandFilterOptions](@ref sfFDN::ThreeBandFilterOptions) adds a middle band to the attenuation design. It uses two shelf frequencies and three T60 targets to shape decay over low, mid, and high frequency ranges. The resulting filter is composed of a 2nd-order low-shelf filter and a 2nd-order high-shelf filter.

The filter is designed by [sfFDN::DesignThreeBandAbsorption](@ref sfFDN::DesignThreeBandAbsorption), which returns a cascade of biquad sections suitable for direct use in the loop filter path.

Key fields:

- `t60s`: Three target T60 values for low, mid, and high bands.
- `delay`: Delay in samples used by the design equation. If this value if <= 0, sfFDN will automatically update it from the surrounding FDN configuration.
- `freqs`: Shelf crossover frequencies that separate the three bands.
- `q`: Shelf Q factor used in the filter design.
- `sample_rate`: Sampling rate used during design.

### Ten-Band Filter

[sfFDN::TenBandFilterOptions](@ref sfFDN::TenBandFilterOptions) provides the most detailed attenuation control in the public API. It targets ten octave-style bands and is implemented as a cascade of second-order biquad sections.

The filter is designed by [sfFDN::DesignTenBandAbsorption](@ref sfFDN::DesignTenBandAbsorption), which follows the two-stage attenuation filter method described in [2].

Key fields:

- `t60s`: Ten target T60 values, one per band.
- `delay`: Delay in samples used by the design equation. If this is non-positive, sfFDN updates it from the surrounding FDN configuration.
- `sample_rate`: Sampling rate used during design.
- `shelf_cutoff`: Shelf crossover used by the design procedure.

## Design Helpers

To create a single filter from one of the variants, the sfFDN::CreateAttenuationFilter function can be used. In that case, the `delay` field of the option struct must be set to a valid value.

```c++
sfFDN::TwoBandFilterOptions options;
options.t60s = { 1.0f, 0.5f };
options.delay = 1007;
options.sample_rate = 48000.0f;

auto filter = sfFDN::CreateAttenuationFilter(options);
```

To create a parallel bank of filters, use sfFDN::CreateAttenuationFilterBank.

```c++
sfFDN::AttenuationFilterBankOptions options;

sfFDN::TwoBandFilterOptions options1;
options1.t60s = {1.0f, 0.5f};
options1.delay = 1007;
options1.sample_rate = 48000.0f;
options.filter_configs.emplace_back(options1);

sfFDN::TwoBandFilterOptions options2;
options2.t60s = {1.0f, 0.5f};
options2.delay = 1433;
options2.sample_rate = 48000.0f;
options.filter_configs.emplace_back(options2);

auto filter_bank = sfFDN::CreateAttenuationFilterBank(options);
```

A version of this function that accepts a single sfFDN::attenuation_filter_variant_t and a list of delay lengths is also provided for convenience. This can be useful when the same filter design is desired across all delay lines, with the only difference being the delay length used in the design equation.

## References

[1]Jot, J.-M., & Chaigne, A. (1991). Digital delay networks for designing artificial reverberators. Proceedings of the 90th Audio Engineering Society Convention (AES), (3030), 1–16.

[2] Välimäki, V., Prawda, K., & Schlecht, S. J. (2024). Two-Stage Attenuation Filter for Artificial Reverberation. IEEE Signal Processing Letters, 31, 391–395. https://doi.org/10.1109/LSP.2024.3352510
