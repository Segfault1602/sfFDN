#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <numbers>
#include <optional>
#include <variant>
#include <vector>

namespace sfFDN
{

// helper type for the visitor #4
template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

constexpr uint32_t kDefaultSampleRate = 48000;
constexpr uint32_t kDefaultBlockSize = 128;

/** @defgroup AudioProcessorOptions Audio Processors Options
 * @brief Structs for configuring audio processors used in the FDN.
 */

/** \addtogroup AudioProcessorOptions
 *  @{
 */

// ENUMS

/** @brief Represents the type of a scalar matrix.
 *
 * [1] D. Rocchesso and J. O. Smith, “Circulant and elliptic feedback delay networks for artificial reverberation,” IEEE
 Transactions on Speech and Audio Processing, vol. 5, no. 1, pp. 51–63, Jan. 1997, doi: 10.1109/89.554269.\n
* [2] S. J. Schlecht, “FDNTB: the feedback delay network toolbox,” 23rd International Conference on Digital Audio
Effects (DAFx2020), 2020.\n
* [3] O. Das, E. K. Canfield-Dafilou, and J. S. Abel, “On The Behavior of Delay Network Reverberator Modes,” in 2019
IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), Oct. 2019, pp. 50–54.
doi: 10.1109/WASPAA.2019.8937260.
*/
enum class ScalarMatrixType : uint8_t
{
    Identity = 0,          /**< Identity matrix. */
    Random = 1,            /**< Random orthogonal matrix. */
    Householder = 2,       /**< Householder matrix. */
    RandomHouseholder = 3, /**< Random Householder matrix. */
    Hadamard = 4,          /**< Hadamard matrix. */
    Circulant = 5,         /**< Circulant matrix as described in [1] */
    Allpass = 6,           /**< Allpass matrix. See [2]*/
    NestedAllpass = 7,     /**< Nested Allpass matrix. See [2] */
    VariableDiffusion = 8, /**< Variable diffusion matrix as described in [3] */
    Count = 9
};

/** @brief Types of interpolation for fractional delay lengths. */
enum class DelayInterpolationType : uint8_t
{
    //! No interpolation. The delay length will be rounded to the nearest integer value.
    None = 0,

    //! Linear interpolation.
    Linear = 1,

    //! Allpass interpolation.
    Allpass = 2,

    // Lagrange interpolation.
    Lagrange = 3,
    Count = 4,
};

/**
 * @brief Types of delay length distributions.
 */
enum class DelayLengthType : uint8_t
{
    //! Delay lengths are generated randomly within the specified range based on a uniform distribution
    Random = 0,

    //! Delay lengths are generated based on a Gaussian distribution within the specified range.
    Gaussian = 1,

    //! Delay lengths are selected randomly from a list of prime numbers within the specified range.
    Primes = 2,

    //! Delay lengths are uniformly distributed within the specified range.
    Uniform = 3,

    //! Delay lengths are generated as powers of prime numbers within the specified range.
    //! Based on https://ccrma.stanford.edu/~jos/pasp/Prime_Power_Delay_Line_Lengths.html
    PrimePower = 4,

    //! Delay lengths are generated using the algorithm from the SteamAudio library.
    SteamAudio = 5,

    Count = 6,
};

/** @brief Enumeration for parallel gain processing modes. */
enum class ParallelGainsMode : uint8_t
{
    //! Process input as a single channel and output to multiple channels
    Split,

    //! Process each input channel separately and output to one channel
    Merge,

    //! Process each input channel separately and output to the same number of channels
    Parallel
};

// STRUCTS

/** @brief Options for configuring a scalar feedback matrix.
 *
 * Can be use to construct a ScalarFeedbackMatrix.
 */
struct ScalarFeedbackMatrixOptions
{
    //! Size of the feedback matrix
    uint32_t matrix_size;

    //! Type of the feedback matrix
    ScalarMatrixType type{ScalarMatrixType::Random};

    //!   Optional custom matrix values in col-major order. The size of the vector must be equal to
    //! matrix_size*matrix_size. If this is set, `type` is ignored.
    std::optional<std::vector<float>> custom_matrix{std::nullopt};

    //! Optional. Seed for random number generation when type is Random or RandomHouseholder.
    uint32_t rng_seed{0};

    //! Optional argument for certain matrix types. For example, for the VariableDiffusion type, this could represent
    // the diffusion parameter.
    std::optional<float> arg{std::nullopt};
};

/** @brief Information structure for constructing a cascaded feedback matrix (also known as a filter feedback matrix).
 *
 * A cascaded feedback matrix is composed of multiple stages, where each stage consists of a scalar feedback matrix
 * followed by a bank of delay lines.
 *
 * Can be used to construct a FilterFeedbackMatrix.
 */
struct CascadedFeedbackMatrixOptions
{
    uint32_t matrix_size{0}; /**< Size of the feedback matrix */
    uint32_t stage_count{0}; /**< Number of stages */
    float sparsity{1.f};     /**< Sparsity level (>= 1). A value of 1 corresponds to a fully dense matrix, while higher
                                values correspond to sparser matrices. */
    ScalarMatrixType type{
        ScalarMatrixType::Random}; /**< Type of the feedback matrix. The same type is used for all stages. */
    float gain_per_samples{1.f};   /**< Gain per sample. */
};

/** @brief Options for configuring signal modulation. */
struct ModulationOptions
{
    float frequency{0.f};     /*< Frequency of the modulation, normalized to [0, 1] by the sampling rate. */
    float amplitude{0.f};     /*< Amplitude of the modulation. */
    float initial_phase{0.f}; /*< Initial phase of the modulation, normalized to [0, 1]. */
};

/** @brief Options for configuring parallel gain processing. */
struct ParallelGainsOptions
{
    ParallelGainsMode mode{ParallelGainsMode::Split}; /**< Mode of parallel gain processing. */
    std::vector<float>
        gains; /**< Gain values for each channel. The size of the vector determines the number of channels. */
    std::vector<ModulationOptions>
        time_varying_config; /**< Optional time-varying modulation configuration for each channel. The size of the
                                  vector must match the size of `gains` if provided. */
};

/** @brief Options for configuring delays. */
struct DelayOptions
{
    float delay{256.f};      /*< Delay in samples. This can be a fractional value if interpolation is used. */
    uint32_t max_delay{512}; /*< Maximum delay in samples. This is used to determine the size of the delay buffer and
                             must be greater than or equal to `delay`. */
    sfFDN::DelayInterpolationType interp_type{
        sfFDN::DelayInterpolationType::None}; /*< Interpolation type for fractional delays. */
    std::optional<sfFDN::ModulationOptions> lfo_config{
        std::nullopt}; /*< Optional LFO configuration for time-varying delay modulation. If provided, the delay will be
                          modulated according to the specified parameters. */
};

/** @brief Options for configuring a delay bank. */
struct DelayBankOptions
{
    std::vector<float>
        delays; /*< Delay values for each channel in samples. These can be fractional values if interpolation is used.
              The size of the vector determines the number of channels in the delay bank. */
    uint32_t block_size{kDefaultBlockSize}; /*< Block size for processing audio. This is used to determine the size of
                                               internal buffers and can affect performance. */
    DelayInterpolationType interpolation_type{
        DelayInterpolationType::None}; /*< Interpolation type for fractional delays. */
};

/** @brief Options for configuring a time-varying delay bank. */
struct DelayBankTimeVaryingOptions
{
    std::vector<float> delays; /*< Initial delay values for each channel in samples. These can be fractional values if
              interpolation is used. The size of the vector determines the number of channels in the delay bank. */
    uint32_t max_delay;        /*< Maximum delay in samples. This is used to determine the size of the delay buffer and
                                    must be greater than or equal to the initial delays. */
    DelayInterpolationType interpolation_type;          /*< Interpolation type for fractional delays. */
    std::vector<ModulationOptions> time_varying_config; /*< Time-varying modulation configuration for each channel. The
                                                           size of the vector must match the size of `delays`. */
};

/** @brief Coefficients for a digital IIR filter. */
struct FilterCoefficients
{
    //! Feedforward coefficients
    float b0;

    //! Feedforward coefficients
    float b1;

    //! Feedforward coefficients
    float b2;

    //! Feedback coefficient
    float a0;

    //! Feedback coefficient
    float a1;

    //! Feedback coefficient
    float a2;

    /** @brief Returns the filter coefficients normalized so that a0 is equal to 1. */
    FilterCoefficients Normalize() const
    {
        return {.b0 = b0 / a0, .b1 = b1 / a0, .b2 = b2 / a0, .a0 = 1.0f, .a1 = a1 / a0, .a2 = a2 / a0};
    }
};

/** @brief Options for configuring an allpass filter. */
struct AllpassFilterOptions
{
    /** @brief The coefficient for the allpass filter. */
    float coeff{0.f};
};

/** @brief Options for configuring a sparse FIR filter. */
struct SparseFirOptions
{
    std::vector<std::pair<uint32_t, float>> coeffs; // pair of (index, coefficient)
};

/** @brief Options for configuring cascaded biquad filters. */
struct CascadedBiquadsOptions
{
    std::vector<FilterCoefficients> coeffs;
};

/** @brief Options for configuring a FIR filter. */
struct FirOptions
{
    std::vector<float> coeffs{1.f};
};

struct MultichannelFirOptions
{
    std::vector<std::vector<float>> coeffs;
};

/** @brief Options for configuring a Schroeder allpass section consisting of `N` Schroeder allpass in series or in
 * parallel.*/
struct SchroederAllpassSectionOptions
{
    std::vector<float> delays; /*< Initial delay values for each Schroeder allpass in samples. */
    std::vector<float> gains;  /*< Feedback gain values for each Schroeder allpass. The size of this vector must match
                                  the size of `delays`. */
    bool parallel{false}; /*< If true, the allpass filters in the section are connected in parallel. If false, they are
                             connected in series. */
};

/** @brief Options for configuring a multichannel bank of Schroeder allpass sections. Each section processes one channel
 * of audio. */
struct MultichannelSchroederAllpassSectionOptions
{
    std::vector<SchroederAllpassSectionOptions> sections;
};

/** @brief Options for configuring a homogenous filter. The homogenous filter has the same attenuation characteristics
 * across all frequencies. */
struct HomogenousFilterOptions
{
    float t60 = 1.f;  /*< Target T60 value for the filter. */
    float delay{1.f}; /*< Delay in samples for the delay line preceding the filter. If set to <= 0, it will be updated
                    automatically when accessed from `CreateFDNFromConfig()`*/
    float sample_rate = kDefaultSampleRate; /*< Sample rate in Hz. This is used to calculate the filter coefficients
                                               based on the specified T60 values. */
};

/** @brief Options for configuring a two-band filter. The two-band filter allows for specifying a target t60 at DC and
 * Nyquist. The filter is implemented as a one-pole filter based on [1].
 *
 * [1] Jot, J. M., & Chaigne, A. (1991). Digital delay networks for designing artificial reverberators (pp. 1-12).
 * Presented at the Proc. Audio Eng. Soc. Conv., Paris, France.
 */
struct TwoBandFilterOptions
{
    std::array<float, 2> t60s{1.f, 0.5f}; /**< Target T60 values for the low and high bands. */
    float delay{0.f}; /*< Delay in samples for the delay line preceding the filter. If set to <= 0, it will be updated
                    automatically when accessed from `CreateFDNFromConfig()`*/
    float sample_rate = kDefaultSampleRate; /*< Sample rate in Hz. This is used to calculate the filter coefficients
                                               based on the specified T60 values. */
};

/** @brief Options for configuring a three-band filter. The three-band filter is composed of a 2nd order low shelf and a
 * 2nd order high shelf filter in series and allows control of the T60 over three bands with the first band being [0,
 * freq[0]], the second band [freq[0], freq[1]], and the third band [freq[1], Nyquist]. */
struct ThreeBandFilterOptions
{
    std::array<float, 3> t60s{1.f, 0.5f, 0.25f}; /*< Target T60 values for the low, mid and high bands. */
    float delay{0.f}; /*< Delay in samples for the delay line preceding the filter. If set to <= 0, it will be updated
                    automatically when accessed from `CreateFDNFromConfig()`*/
    std::array<float, 2> freqs{800.f, 8000.f};    /*< Frequency values for the low and high shelves. */
    float q = 1.f / std::numbers::sqrt2_v<float>; /*< Q-factor for the shelf filters. Q values higher than 0.707 may
                                                     cause instability if placed in a feedback loop. */
    float sample_rate = kDefaultSampleRate; /*< Sample rate in Hz. This is used to calculate the filter coefficients
                                               based on the specified T60 values. */
};

/** @brief Options for configuring a ten-band filter. The ten-band filter allows control of the T60 over ten bands.
 * The bands of the filter are set to {32, 64, 125, 250, 500, 1k, 2k, 4k, 8k, 16k} Hz.
 *
 * The filter is implemented as a cascade of second-order biquad filters following the design method described in [1].
 * [1] V. Välimäki, K. Prawda, and S. J. Schlecht, "Two-Stage Attenuation Filter for Artificial Reverberation,"
 * IEEE Signal Processing Letters, vol. 31, pp. 391–395, 2024, doi: 10.1109/LSP.2024.3352510.
 */
struct TenBandFilterOptions
{
    //! Target T60 values for the ten bands.
    std::array<float, 10> t60s = {1.f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};

    //! Delay in samples for the delay line preceding the filter. If set to <= 0, it will be updated automatically when
    //! accessed from `CreateFDNFromConfig()`
    float delay{0.f};

    //! Sample rate in Hz. This is used to calculate the filter coefficients based on the specified T60 values.
    float sample_rate = kDefaultSampleRate;

    //! Cutoff frequency for the shelf filters.
    float shelf_cutoff = 8000.f;
};

/** @brief Variant type for holding different attenuation filter options. */
using attenuation_filter_variant_t =
    std::variant<HomogenousFilterOptions, TwoBandFilterOptions, ThreeBandFilterOptions, TenBandFilterOptions>;

/** @brief Options for configuring an attenuation filter bank. */
struct AttenuationFilterBankOptions
{
    //! Vector of attenuation filter configurations.
    std::vector<attenuation_filter_variant_t> filter_configs;
};

/** @brief Options for configuring a graphic equalizer. */
struct GraphicEQOptions
{
    //! Target gains for the ten bands in dB.
    std::array<float, 10> gains_db;

    //! Frequency values for the ten bands in Hz.
    std::array<float, 10> freqs;

    //! Sample rate in Hz.
    float sample_rate = kDefaultSampleRate;
};

/** @brief Variant type for holding different feedback matrix options. */
using feedback_matrix_variant_t = std::variant<CascadedFeedbackMatrixOptions, ScalarFeedbackMatrixOptions>;

/** @brief Variant type for holding different single-channel processor options. */
using single_channel_processor_variant_t =
    std::variant<SchroederAllpassSectionOptions, AllpassFilterOptions, CascadedBiquadsOptions, FirOptions, DelayOptions,
                 GraphicEQOptions>;

/** @brief Variant type for holding different multi-channel processor options. */
using multi_channel_processor_variant_t =
    std::variant<ParallelGainsOptions, MultichannelSchroederAllpassSectionOptions, AttenuationFilterBankOptions,
                 DelayBankOptions, DelayBankTimeVaryingOptions, CascadedFeedbackMatrixOptions,
                 ScalarFeedbackMatrixOptions, MultichannelFirOptions>;

/** @}*/

NLOHMANN_JSON_SERIALIZE_ENUM(ScalarMatrixType, {{ScalarMatrixType::Identity, "Identity"},
                                                {ScalarMatrixType::Random, "Random"},
                                                {ScalarMatrixType::Householder, "Householder"},
                                                {ScalarMatrixType::RandomHouseholder, "RandomHouseholder"},
                                                {ScalarMatrixType::Hadamard, "Hadamard"},
                                                {ScalarMatrixType::Circulant, "Circulant"},
                                                {ScalarMatrixType::Allpass, "Allpass"},
                                                {ScalarMatrixType::NestedAllpass, "NestedAllpass"},
                                                {ScalarMatrixType::VariableDiffusion, "VariableDiffusion"},
                                                {ScalarMatrixType::Count, "Count"}});

NLOHMANN_JSON_SERIALIZE_ENUM(DelayInterpolationType, {{DelayInterpolationType::None, "None"},
                                                      {DelayInterpolationType::Linear, "Linear"},
                                                      {DelayInterpolationType::Allpass, "Allpass"},
                                                      {DelayInterpolationType::Lagrange, "Lagrange"}});

NLOHMANN_JSON_SERIALIZE_ENUM(DelayLengthType, {{DelayLengthType::Random, "Random"},
                                               {DelayLengthType::Gaussian, "Gaussian"},
                                               {DelayLengthType::Primes, "Primes"},
                                               {DelayLengthType::Uniform, "Uniform"},
                                               {DelayLengthType::PrimePower, "PrimePower"},
                                               {DelayLengthType::SteamAudio, "SteamAudio"}});

NLOHMANN_JSON_SERIALIZE_ENUM(ParallelGainsMode, {{ParallelGainsMode::Split, "Split"},
                                                 {ParallelGainsMode::Merge, "Merge"},
                                                 {ParallelGainsMode::Parallel, "Parallel"}});

void to_json(nlohmann::json& j, const ScalarFeedbackMatrixOptions& config);
void from_json(const nlohmann::json& j, ScalarFeedbackMatrixOptions& config);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CascadedFeedbackMatrixOptions, matrix_size, stage_count, sparsity, type,
                                   gain_per_samples);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ModulationOptions, frequency, amplitude, initial_phase);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ParallelGainsOptions, mode, gains, time_varying_config);
void to_json(nlohmann::json& j, const DelayOptions& config);
void from_json(const nlohmann::json& j, DelayOptions& config);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DelayBankOptions, delays, block_size, interpolation_type);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DelayBankTimeVaryingOptions, delays, max_delay, interpolation_type,
                                   time_varying_config);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FilterCoefficients, b0, b1, b2, a0, a1, a2);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AllpassFilterOptions, coeff);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SparseFirOptions, coeffs);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CascadedBiquadsOptions, coeffs);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FirOptions, coeffs);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MultichannelFirOptions, coeffs);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SchroederAllpassSectionOptions, delays, gains, parallel);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MultichannelSchroederAllpassSectionOptions, sections);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(HomogenousFilterOptions, t60, delay, sample_rate);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TwoBandFilterOptions, t60s, delay, sample_rate);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ThreeBandFilterOptions, t60s, delay, freqs, q, sample_rate);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TenBandFilterOptions, t60s, delay, sample_rate, shelf_cutoff);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GraphicEQOptions, gains_db, freqs, sample_rate);

void to_json(nlohmann::json& j, const AttenuationFilterBankOptions& config);
void from_json(const nlohmann::json& j, AttenuationFilterBankOptions& config);

} // namespace sfFDN