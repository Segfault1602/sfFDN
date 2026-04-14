#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <numbers>
#include <optional>
#include <tuple>
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

enum class DelayInterpolationType : uint8_t
{
    None,
    Linear,
    Allpass,
    Lagrange,
};

/**
 * @brief Types of delay length distributions.
 */
enum class DelayLengthType : uint8_t
{

    Random = 0,     /**< %Delay lengths are generated randomly within the specified range based on a uniform
                       distribution. */
    Gaussian = 1,   /**< %Delay lengths are generated based on a Gaussian distribution within the specified range. */
    Primes = 2,     /**< %Delay lengths are selected randomly from a list of prime numbers. */
    Uniform = 3,    /**< %Delay lengths are uniformly distributed within the specified range. */
    PrimePower = 4, /**< %Delay lengths are generated as powers of prime numbers within the specified range.
     Based on https://ccrma.stanford.edu/~jos/pasp/Prime_Power_Delay_Line_Lengths.html*/
    SteamAudio = 5, /**< %Delay lengths are generated using the algorithm from the SteamAudio library. */

    Count = 6,
};

/** @brief Enumeration for parallel gain processing modes. */
enum class ParallelGainsMode : uint8_t
{
    Split,   /** < Process input as a single channel and output to multiple channels */
    Merge,   /** < Process each input channel separately and output to one channel */
    Parallel /** < Process each input channel separately and output to the same number of channels */
};

// STRUCTS

struct ScalarFeedbackMatrixOptions
{
    uint32_t matrix_size;
    ScalarMatrixType type{ScalarMatrixType::Random};
    std::optional<std::vector<float>> custom_matrix{std::nullopt};

    uint32_t rng_seed{0};
    std::optional<float> arg{std::nullopt};
};

/** @brief Information structure for constructing a cascaded feedback matrix (also known as a filter feedback matrix).
 */
struct CascadedFeedbackMatrixOptions
{
    uint32_t matrix_size;                     /**< Size of the feedback matrix */
    uint32_t stage_count;                     /**< Number of stages */
    std::vector<std::vector<float>> delays;   /**< Delays, size: stage_count x N */
    std::vector<std::vector<float>> matrices; /**< Feedback matrices, size: K x N x N */
};

struct ModulationOptions
{
    float frequency{0.f};
    float amplitude{0.f};
    float initial_phase{0.f};
};

struct ParallelGainsOptions
{
    ParallelGainsMode mode{ParallelGainsMode::Split};
    std::vector<float> gains;
    std::vector<ModulationOptions> time_varying_config{};
};

struct DelayOptions
{
    float delay{256.f};
    uint32_t max_delay{512};
    sfFDN::DelayInterpolationType interp_type{sfFDN::DelayInterpolationType::Allpass};
    std::optional<sfFDN::ModulationOptions> lfo_config{std::nullopt};
};

struct DelayBankOptions
{
    std::vector<float> delays;
    uint32_t block_size{kDefaultBlockSize};
    DelayInterpolationType interpolation_type{DelayInterpolationType::None};
};

struct DelayBankTimeVaryingOptions
{
    std::vector<float> delays;
    uint32_t max_delay;
    DelayInterpolationType interpolation_type;
    std::vector<ModulationOptions> time_varying_config;
};

struct FilterCoefficients
{
    float b0, b1, b2, a0, a1, a2;

    FilterCoefficients Normalize() const
    {
        return {b0 / a0, b1 / a0, b2 / a0, 1.0f, a1 / a0, a2 / a0};
    }
};

struct AllpassFilterOptions
{
    float coeff{0.f};
};

struct SparseFirOptions
{
    std::vector<std::pair<uint32_t, float>> coeffs; // pair of (index, coefficient)
};

struct CascadedBiquadsOptions
{
    std::vector<FilterCoefficients> coeffs;
};

struct FirOptions
{
    std::vector<float> coeffs{1.f};
};

struct SchroederAllpassSectionOptions
{
    std::vector<uint32_t> delays;
    std::vector<float> gains;
    bool parallel{false};
};

struct ParallelSchroederAllpassSectionOptions
{
    std::vector<SchroederAllpassSectionOptions> sections;
};

struct ProportionalAttenuationOptions
{
    float t60 = 1.f;
    float delay;
    float sample_rate = kDefaultSampleRate;
};

struct TwoBandFilterOptions
{
    std::array<float, 2> t60s{1.f, 0.5f};
    float delay;
    float sample_rate = kDefaultSampleRate;
};

struct ThreeBandFilterOptions
{
    std::array<float, 3> t60s{1.f, 0.5f, 0.25f};
    float delay;
    std::array<float, 2> freqs{800.f, 8000.f};
    float q = 1.f / std::numbers::sqrt2_v<float>;
    float sample_rate = kDefaultSampleRate;
};

struct TenBandFilterOptions
{
    std::array<float, 10> t60s = {1.f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    float delay;
    float sample_rate = kDefaultSampleRate;
    float shelf_cutoff = 8000.f;
};

using attenuation_filter_variant_t =
    std::variant<ProportionalAttenuationOptions, TwoBandFilterOptions, ThreeBandFilterOptions, TenBandFilterOptions>;

struct AttenuationFilterBankOptions
{
    std::vector<attenuation_filter_variant_t> filter_configs;
};

struct GraphicEQOptions
{
    std::array<float, 10> gains_db;
    std::array<float, 10> freqs;
    float sample_rate = kDefaultSampleRate;
};

using feedback_matrix_variant_t = std::variant<CascadedFeedbackMatrixOptions, ScalarFeedbackMatrixOptions>;
using single_channel_processor_variant_t =
    std::variant<SchroederAllpassSectionOptions, AllpassFilterOptions, CascadedBiquadsOptions, FirOptions, DelayOptions,
                 GraphicEQOptions>;

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
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CascadedFeedbackMatrixOptions, matrix_size, stage_count, delays, matrices);
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
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SchroederAllpassSectionOptions, delays, gains, parallel);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ParallelSchroederAllpassSectionOptions, sections);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ProportionalAttenuationOptions, t60, delay, sample_rate);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TwoBandFilterOptions, t60s, delay, sample_rate);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ThreeBandFilterOptions, t60s, delay, freqs, q, sample_rate);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TenBandFilterOptions, t60s, delay, sample_rate, shelf_cutoff);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GraphicEQOptions, gains_db, freqs, sample_rate);

void to_json(nlohmann::json& j, const AttenuationFilterBankOptions& config);
void from_json(const nlohmann::json& j, AttenuationFilterBankOptions& config);

} // namespace sfFDN