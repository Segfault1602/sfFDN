# sfFDN: Real-Time Feedback Delay Network Library

**sfFDN** is a C++ library inspired by the MATLAB Feedback Delay Network Toolbox (FDNTB) by S. J. Schlecht[^1]. It provides efficient implementations of FDNs with various features such as:
- Configurable delay lines
- Different types of feedback matrices (e.g., Hadamard, Householder, Random, Circulant, etc.)
- Filter Feedback Matrices (FFM) as presented in [^2]
- Single channel and multi-channel IIR filters
- Attenuation filter variants for frequency-dependent decay control
- Graphic Equalizers filter as presented in [^3]
- Partitioned convolution for FIR filtering
- Sparse FIR filter
- Schroeder all-pass filters
- Time-varying delay lines
- Time-varying input and output gains

## Architecture
The FDN topology implemented in sfFDN is based on the canonical structure found in the literature and is shown here:
![sfFDN Architecture](../../sfFDN.svg)

This topology can be separated into seven building blocks: the input gains (green), the delay lines (yellow), the loop filters (red), the feedback matrix (orange), the output gains (blue), the tone correction filter (purple), and the direct gain (gray). At the heart of the library is the *AudioProcessor* interface. In the context of sfFDN, an audio processor is defined as a class that can take \f$N_{in}\f$ channels of audio, apply a transformation (e.g., filtering, delay, mixing matrix), and finally output \f$N_{out}\f$ channels of audio. The `AudioProcessorChain` class can be used to chain multiple audio processors in series and the `FilterBank` class can similarly be used to group multiple single-channel audio processors into a bank of parallel processor.

<details>

<summary> Input/Output gains </summary>

The input gains block supports any processor that takes a single channel of audio as input and outputs \f$N\f$ channels of audio. Conversely, the output gains block consists of any processor that takes \f$N\f$ channels of audio as input and outputs a single channel of audio. The simplest and most common implementation applies one scalar gain (\f$b_i\f$, \f$c_i\f$) per delay line. `FDN::SetInputGains(span)`, `FDN::SetOutputGains(span)`, and unmodulated `StageGainsOptions` keep that compact gain-list interface but construct a `ChannelMatrix` internally. `ParallelGains` remains available as a standalone split, merge, or diagonal processor, and `TimeVaryingParallelGains` implements modulated stage gains. FIR filters are commonly added at the input and/or output of the FDN to simulate early reflections and increase echo density. This effect can be achieved by chaining a `Fir` processor with the boundary routing. For longer FIR filters, `PartitionedConvolver` can reduce convolution cost. Fagerström et al. (2020)[^4] proposed a novel FDN structure where the input and output gains are replaced by velvet noise filters, resulting in an increase in echo density. This so-called velvet-noise FDN can be implemented with `SparseFir`, which efficiently represents sparse FIR filters suited to velvet-noise sequences. `FilterBank` can also create a bank of parallel filters, allowing each channel to have its own FIR or velvet-noise filter.

More generally, the network takes \f$M\f$ external input channels and produces \f$K\f$ external output channels. The input gains block is then the boundary matrix \f$B\f$ mapping \f$M\f$ to \f$N\f$, and the output gains block is \f$C\f$ mapping \f$N\f$ to \f$K\f$; `ChannelMatrix` implements both. \f$M\f$, \f$N\f$ and \f$K\f$ are fixed for the lifetime of an `FDN` and are supplied at construction through `FDNTopology`:

```c++
sfFDN::FDN fdn(sfFDN::FDNTopology{
    .order = 8,
    .block_size = 128,
    .input_channel_count = 1,
    .output_channel_count = 2,
});
```

Every processor installed afterwards is validated against those counts, so a setter either installs a compatible processor or leaves the network untouched; there is no way to resize an existing FDN. `FDN::Process` requires an input buffer with exactly \f$M\f$ channels. It requires exactly \f$K\f$ output channels, except that a network with \f$K = 1\f$ also accepts a wider output buffer and duplicates channel zero into the rest. `FDN::Process` overwrites its output buffer, so callers driving it block by block do not have to clear the destination; any previous contents are discarded rather than mixed into the response.


</details>

<details>

<summary> Delay Lines </summary>

Primary delay lengths are in samples and must be at least the FDN's processing block size.
Use `DelayBankOptions::interpolation_type` or the interpolation argument to `FDN::SetDelays()`
for fractional lengths. `DelayTimeVarying` processors can add modulated delays inside the loop.
`GetDelayLengths()` generates delay lengths using several heuristics:

- **Random** Randomly generate delay lengths pulled from a uniform distribution
- **Gaussian**: Randomly generate delay lengths pulled from a Gaussian distribution.
- **Prime**: Randomly generate delay lengths pulled from a uniform distribution. Delays are guaranteed to be prime numbers.
- **Uniform**: Delays are uniformly spaced between a minimum and maximum value.
- **Prime Power**: Delays are integer powers of prime numbers. Based on the implementation found
in the Faust library
- **Steam Audio**: Re-implementation of the delay length generation method used in the reverberator of the Steam Audio Library. This heuristic is based on the Prime Power method but with some amount of randomization added.
- **Mean Delay**: Delays are generated based on Eq.(40) from (Schlecht & Habets, 2017a)[^5]. The resulting delays are logarithmically spaced to obtain a desired mean delay length and standard
deviation.

</details>

<details>
<summary> Feedback Matrix </summary>

The feedback matrix supports any processor that takes N channels of audio as input and outputs \f$N\f$ channels of audio. Common feedback matrices implemented in sfFDN by the ScalarFeedbackMatrix class include the Hadamard, Householder, random orthogonal, circulant, the allpass and nested allpass feedback matrix from (Schlecht, 2021)[^6], as well as the identity matrix. A scalar matrix is either a generated recipe or explicit `MatrixData`; the latter owns its coefficients. The FilterFeedbackMatrix class is also provided and implements the filter feedback matrix structure proposed by Schlecht and Habets (2020)[^7].

</details>

## Matrix coefficient order and pyFDN

Public matrix vectors use row-major destination/output-row, source/input-column order:
`matrix[row * N + column] = A[row, column]`, and the processor applies \f$y = A x\f$. To pass a
pyFDN/NumPy matrix `A` with shape `(out, in)`, use its C-order flattening:

```python
coefficients = numpy.asarray(A).ravel(order="C")
```

pyFDN evaluates the same mapping as `x @ A.T`. This is unrelated to sfFDN's **transposed FDN
topology** (`FDN::SetTranspose`): that setting reorders the FDN signal-flow topology; it does not
apply \f$A^T\f$ to a supplied feedback matrix. Supply an explicitly transposed matrix when
\f$A^T\f$ is desired.

## Matrix sources and seeds

`ScalarFeedbackMatrixOptions::source` is either a generated `GeneratedMatrixOptions` recipe or
owned, shape-checked `MatrixData`. `MatrixData(order, coefficients)` verifies that its row-major
coefficient vector has exactly `order * order` values; use its `Values()` spans to inspect or edit
those owned values.

```c++
// Generated recipe
sfFDN::ScalarFeedbackMatrixOptions generated{
    .source = sfFDN::GeneratedMatrixOptions{
        .matrix_size = kFDNOrder,
        .generator = sfFDN::ScalarMatrixType::Hadamard,
        .rng_seed = sfFDN::kDefaultMatrixSeed}};

// Explicit row-major data for y = A x
sfFDN::ScalarFeedbackMatrixOptions explicit_matrix{
    .source = sfFDN::MatrixData{2, {1.f, 0.f, 0.f, 1.f}}};

// Parameterized Variable Diffusion recipe
sfFDN::ScalarFeedbackMatrixOptions diffusion{
    .source = sfFDN::GeneratedMatrixOptions{
        .matrix_size = kFDNOrder,
        .generator = sfFDN::VariableDiffusionOptions{.diffusion = 0.5f},
        .rng_seed = 0U}};
```

<details>
<summary> Loop Filters </summary>

The optional loop-filter block takes \f$N\f$ channels of audio and outputs \f$N\f$ channels.
`CreateAttenuationFilterBank()` builds decay-control filters from attenuation options.
Choose `HomogenousFilterOptions` for frequency-independent decay, or `TwoBandFilterOptions`,
`ThreeBandFilterOptions`, or `TenBandFilterOptions` for frequency-dependent T60 targets.
Pass one design with a span of delay lengths, or an `AttenuationFilterBankOptions` value
containing per-channel designs. `FilterBank` also supports arbitrary parallel single-channel processors.

</details>

## Example Usage

Here is an example of how to create a 'classic' FDN of 8 delay lines with a Hadamard feedback matrix:

```c++
#include <sffdn/sffdn.h>
constexpr uint32_t kSampleRate = 48000;
constexpr uint32_t kFDNOrder = 8;

constexpr uint32_t kBlockSize = 128;
sfFDN::FDN fdn(kFDNOrder, kBlockSize);

// Set all input gains to 0.5
const std::vector<float> input_gains(kFDNOrder, 0.5f);
fdn.SetInputGains(input_gains);

// Set all output gains to 0.5
const std::vector<float> output_gains(kFDNOrder, 0.5f);
fdn.SetOutputGains(output_gains);

// Set Hadamard feedback matrix
sfFDN::ScalarFeedbackMatrixOptions feedback_matrix_options;
feedback_matrix_options.source = sfFDN::GeneratedMatrixOptions{
    .matrix_size = kFDNOrder,
    .generator = sfFDN::ScalarMatrixType::Hadamard};
auto feedback_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(feedback_matrix_options);
fdn.SetFeedbackMatrix(std::move(feedback_matrix));

// Set random delay lengths
const std::vector<float> delays =
    sfFDN::GetDelayLengths(kFDNOrder, 500.f, 3000.f, sfFDN::DelayLengthType::Random);
fdn.SetDelays(delays);

// Set homogeneous decay of 1 second
const sfFDN::HomogenousFilterOptions attenuation_options{
    .t60 = 1.f,
    .delay = 0.f,
    .sample_rate = static_cast<float>(kSampleRate)};
auto attenuation_filter = sfFDN::CreateAttenuationFilterBank(attenuation_options, delays);
fdn.SetLoopFilter(std::move(attenuation_filter));

```

`MakeDefaultFDNConfig()` provides a complete wet configuration for a requested order, block size, and sample rate:

```c++
#include <sffdn/fdn_config.h>
#include <sffdn/fdn.h> // Completes FDN for the returned std::unique_ptr.

auto config = sfFDN::MakeDefaultFDNConfig();
auto smaller_config = sfFDN::MakeDefaultFDNConfig(4U, 64U);
auto fdn = sfFDN::CreateFDNFromConfig(config);
```

It selects deterministic delays in an approximately 20--50 ms range, with each delay at least one block long,
uses normalized input and output gains of `1 / sqrt(N)`, applies one second of homogeneous attenuation, and
chooses Hadamard feedback for power-of-two orders or Householder feedback otherwise.
`FDNConfig{}` is an initialized but invalid empty draft. Configuration equality compares stored members exactly.

`InputStageConfig` and `OutputStageConfig` use `StageGainsOptions` for their compact gain-list representation.
With no modulation, the factory converts those lists to N-by-1 and 1-by-N `ChannelMatrix` processors. A nonempty
modulation vector selects `TimeVaryingParallelGains`; this is based on whether modulation is configured, not on
whether its current amplitude happens to be zero. `ParallelGainsOptions` retains its explicit `mode` for standalone
and multichannel gain processors.

The public config and JSON representation does not change when a static stage is built as a matrix. Likewise,
`FDN::SetInputGains(span)` and `SetOutputGains(span)` remain the recommended convenience APIs. The
`GetInputGains()` and `GetOutputGains()` accessors return `AudioProcessor*`; code must not assume that an
automatically constructed static boundary can be downcast to `ParallelGains`. Install an explicit compatible
processor through the owning setter when its concrete type or accumulating semantics are required.

### Multi-input, multi-output configurations

`FDNConfig::input_channel_count` (M) and `output_channel_count` (K) declare the external shape of the network and
both default to `1`. A count other than `1` requires the corresponding stage to carry an explicit
`boundary_matrix` instead of stage gains:

```c++
config.input_channel_count = 2;
config.output_channel_count = 2;

config.input_block_config.parallel_gains_config = {}; // Must be empty when a boundary matrix is set.
config.input_block_config.boundary_matrix = sfFDN::ChannelMatrixOptions{
    .input_channel_count = config.input_channel_count,
    .output_channel_count = config.fdn_size,
    .coefficients = std::vector<float>(config.input_channel_count * config.fdn_size, 0.5f)};

config.output_block_config.parallel_gains_config = {};
config.output_block_config.boundary_matrix = sfFDN::ChannelMatrixOptions{
    .input_channel_count = config.fdn_size,
    .output_channel_count = config.output_channel_count,
    .coefficients = std::vector<float>(config.output_channel_count * config.fdn_size, 0.5f)};
```

The selection is explicit in both directions and never inferred: a boundary matrix present means the matrix is
used and the stage gains must be empty; a boundary matrix absent means the stage gains are used and the
corresponding channel count must be `1`.

The direct path follows the same rule. `direct_gain` is a diagonal `gain * I` and therefore requires M to equal K;
set `direct_matrix` for any other shape. A nonzero `direct_gain` together with a `direct_matrix` is rejected as
ambiguous.

Two placement rules follow from the topology. `single_channel_processors` on the input stage run *before* the
input matrix and on the output stage run *after* the output matrix, so they sit on the external side of the
boundary. With M or K greater than 1 the whole ordered chain is replicated once per external channel, each replica
an independent instance with its own filter state. Tone correction follows the same rule on the K output channels.


Include `<sffdn/serialization.h>` to serialize `FDNConfig` to JSON and link JSON consumers to
`sfFDN::serialization`. Reads are transactional: malformed input leaves the destination unchanged.
Tagged processor, matrix, and attenuation-filter wrappers contain exactly one supported type tag.

```c++
sfFDN::FDNConfig config{};
config.fdn_size = 8;
config.transposed = false;
config.direct_gain = 1.f;
config.block_size = 128;
config.sample_rate = 48000;

sfFDN::DelayBankOptions delay_bank_options{
    .delays = sfFDN::GetDelayLengths(config.fdn_size, 500, 3000, sfFDN::DelayLengthType::Random),
    .block_size = config.block_size,
    .interpolation_type = sfFDN::DelayInterpolationType::None};

config.delay_bank_config = delay_bank_options;

sfFDN::StageGainsOptions input_gains_options{
    .gains = std::vector<float>(config.fdn_size, 0.5f)};

config.input_block_config.parallel_gains_config = input_gains_options;

sfFDN::ScalarFeedbackMatrixOptions feedback_matrix_options{
    .source = sfFDN::GeneratedMatrixOptions{
        .matrix_size = config.fdn_size,
        .generator = sfFDN::ScalarMatrixType::Hadamard}};

config.feedback_matrix_config = feedback_matrix_options;

sfFDN::AttenuationFilterBankOptions attenuation_filter_bank_options;
sfFDN::HomogenousFilterOptions homogenous_filter_options{
    .t60 = 1.f,
    .delay = 0.f,
    .sample_rate = config.sample_rate};
attenuation_filter_bank_options.filter_configs.push_back(homogenous_filter_options);

config.attenuation_filter_bank_config = attenuation_filter_bank_options;

sfFDN::StageGainsOptions output_gains_options{
    .gains = std::vector<float>(config.fdn_size, 0.5f)};

config.output_block_config.parallel_gains_config = output_gains_options;

auto fdn = sfFDN::CreateFDNFromConfig(config);
```

The primary delay bank contains one delay per FDN channel. Its block size must be nonzero and at least
`config.block_size`, and each primary delay must also be at least `config.block_size`.
Changing `config.sample_rate` does not update existing delays, modulation frequencies, or nested filter rates;
rate-sensitive filter options use their own `sample_rate` fields.
See [Filtering](filters.md#design-helpers) for attenuation-bank delay inference and placement rules.

### Validating an FDN configuration

`ValidateFDNConfig()` checks graph dimensions and processor-option domains without constructing processors
or modifying the configuration. It reports independent issues as paths and explanations:

```c++
const auto result = sfFDN::ValidateFDNConfig(config);
if (!result) {
    for (const auto& issue : result.error()) {
        Log(issue.path + ": " + issue.message);
    }
}
```

`CreateFDNFromConfig()` performs the same validation and throws `sfFDN::FDNConfigError` for reported issues:

```c++
try {
    auto fdn = sfFDN::CreateFDNFromConfig(config);
} catch (const sfFDN::FDNConfigError& error) {
    for (const auto& issue : error.Issues()) {
        Log(issue.path + ": " + issue.message);
    }
}
```

Successful validation does not guarantee filter-design or matrix-decomposition success, or acoustic stability.

`<sffdn/fdn_config.h>` is sufficient for authoring and validating configurations, but it only
forward-declares `FDN`. Include `<sffdn/fdn.h>` (or `<sffdn/sffdn.h>`) before creating or
destroying the `std::unique_ptr<FDN>` returned by `CreateFDNFromConfig()`.

## Build

The library is built using CMake. **sfFDN** uses [CPM](https://github.com/cpm-cmake/CPM.cmake) to manage dependencies. CMake presets are provided for building with Ninja and LLVM.

```bash
# configure with Ninja and LLVM
cmake --preset llvm-ninja

# build
cmake --build --preset llvm --config Release
```

## Use sfFDN in your project

**sfFDN** can be included in your project using CPM (or CMake's FetchContent directly)
```cmake
CPMAddPackage(
    NAME sfFDN
    GIT_REPOSITORY https://github.com/Segfault1602/sfFDN.git
    GIT_TAG main
    )

target_link_libraries(your_target PRIVATE sfFDN::sfFDN)
```

Core consumers include the public headers they use, for example `<sffdn/fdn_config.h>` for
configuration authoring and `<sffdn/fdn.h>` when they own an `FDN`. They do not receive JSON
headers or a JSON link dependency. A JSON consumer opts in explicitly:

```cmake
target_link_libraries(your_json_target PRIVATE sfFDN::serialization)
```

```c++
#include <sffdn/serialization.h>
```

## Dependencies

- [Eigen](https://eigen.tuxfamily.org/dox/) - Linear algebra library
- [PFFFT](https://bitbucket.org/jpommier/pffft/) - FFT library for partitioned convolution
- [KissFFT](https://github.com/mborgerding/kissfft) - FFT library used for FFT size less than what PFFFT supports
- [nlohmann-json](https://github.com/nlohmann/json) - Used by the opt-in `sfFDN::serialization` target to export/import FDN configurations to JSON files.
- [nanobench](https://github.com/martinus/nanobench) - Microbenchmarking library used for performance testing. Not required if SFFDN_BUILD_TESTS is OFF.
- [Catch2](https://github.com/catchorg/Catch2) - Unit testing framework used for testing. Not required if SFFDN_BUILD_TESTS is OFF.
- [libsndfile](http://www.mega-nerd.com/libsndfile/) - Used in unit tests for reading/writing WAV files. Not required if SFFDN_BUILD_TESTS is OFF.

## References
[^1]: S. J. Schlecht, “FDNTB: the feedback delay network toolbox,” 23rd International Conference on Digital Audio Effects (DAFx2020), 2020.

[^2]: S. J. Schlecht and E. A. P. Habets, “Scattering in Feedback Delay Networks,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, June 2020.

[^3]: V. Välimäki, K. Prawda, and S. J. Schlecht, “Two-Stage Attenuation Filter for Artificial Reverberation,” IEEE Signal Processing Letters, vol. 31, pp. 391–395, Jan. 2024, doi: 10.1109/LSP.2024.3352510.

[^4]: J. Fagerström, B. Alary, S. J. Schlecht, and V. Välimäki, “Velvet-Noise Feedback Delay Network,” in Proc. Int. Conf. Digital Audio Effects (DAFx), 2020.

[^5]: S. J. Schlecht and E. A. P. Habets, “Feedback Delay Networks: Echo Density and Mixing Time,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 25, no. 2, pp. 374–383, Feb. 2017, doi: 10.1109/TASLP.2016.2635027.

[^6]: S. J. Schlecht, “Allpass Feedback Delay Networks,” IEEE Trans. Signal Process., vol. 69, pp. 1028–1038, 2021, doi: 10.1109/TSP.2021.3053507.

[^7]: S. J. Schlecht and E. A. P. Habets, “Scattering in Feedback Delay Networks,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 28, Jun. 2020.
