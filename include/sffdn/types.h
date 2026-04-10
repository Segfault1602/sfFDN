#pragma once

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

struct ScalarFeedbackMatrixConfig
{
    uint32_t matrix_size;
    ScalarMatrixType type{ScalarMatrixType::Random};
    std::optional<std::vector<float>> custom_matrix{std::nullopt};

    uint32_t rng_seed{0};
    std::optional<float> arg{std::nullopt};
};

/** @brief Information structure for constructing a cascaded feedback matrix (also known as a filter feedback matrix).
 */
struct CascadedFeedbackMatrixInfo
{
    uint32_t matrix_size;                     /**< Size of the feedback matrix */
    uint32_t stage_count;                     /**< Number of stages */
    std::vector<std::vector<float>> delays;   /**< Delays, size: stage_count x N */
    std::vector<std::vector<float>> matrices; /**< Feedback matrices, size: K x N x N */
};

struct ModulationConfig
{
    float frequency;
    float amplitude;
    float initial_phase;
};

struct TimeVaryingParallelGainsConfig
{
    std::vector<float> lfo_frequencies;
    std::vector<float> lfo_amplitudes;
    std::vector<float> lfo_phase_offsets;
};

struct ParallelGainsConfig
{
    ParallelGainsMode mode{ParallelGainsMode::Split};
    std::vector<float> gains;
    std::optional<TimeVaryingParallelGainsConfig> time_varying_config;
};

struct DelayConfig
{
    float delay{256.f};
    uint32_t max_delay{512};
    sfFDN::DelayInterpolationType interp_type{sfFDN::DelayInterpolationType::Allpass};
    std::optional<sfFDN::ModulationConfig> lfo_config{std::nullopt};
};

struct DelayBankConfig
{
    std::vector<float> delays;
    uint32_t block_size{kDefaultBlockSize};
    DelayInterpolationType interpolation_type{DelayInterpolationType::None};
};

struct DelayBankTimeVaryingConfig
{
    std::vector<float> delays;
    uint32_t max_delay;
    DelayInterpolationType interpolation_type;
    std::vector<float> mod_freqs;
    std::vector<float> mod_depths;
    std::vector<float> mod_phase_offsets;
};

struct FilterCoefficients
{
    float b0, b1, b2, a0, a1, a2;

    FilterCoefficients Normalize() const
    {
        return {b0 / a0, b1 / a0, b2 / a0, 1.0f, a1 / a0, a2 / a0};
    }
};

struct AllpassFilterConfig
{
    float coeff{0.f};
};

struct SparseFirConfig
{
    std::vector<std::pair<uint32_t, float>> coeffs; // pair of (index, coefficient)
};

struct CascadedBiquadsConfig
{
    std::vector<FilterCoefficients> coeffs;
};

struct FirConfig
{
    std::vector<float> coeffs;
};

struct SchroederAllpassSectionConfig
{
    std::vector<uint32_t> delays;
    std::vector<float> gains;
    bool parallel{false};
};

struct ParallelSchroederAllpassSectionConfig
{
    std::vector<SchroederAllpassSectionConfig> sections;
};

struct ProportionalAttenuationConfig
{
    float t60 = 1.f;
    float delay;
    float sample_rate = kDefaultSampleRate;
};

struct TwoBandFilterConfig
{
    std::array<float, 2> t60s{1.f, 0.5f};
    float delay;
    float sample_rate = kDefaultSampleRate;
};

struct ThreeBandFilterConfig
{
    std::array<float, 3> t60s{1.f, 0.5f, 0.25f};
    float delay;
    std::array<float, 2> freqs{800.f, 8000.f};
    float q = 1.f / std::numbers::sqrt2_v<float>;
    float sample_rate = kDefaultSampleRate;
};

struct TenBandFilterConfig
{
    std::array<float, 10> t60s = {1.f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    float delay;
    float sample_rate = kDefaultSampleRate;
    float shelf_cutoff = 8000.f;
};

using attenuation_filter_variant_t =
    std::variant<ProportionalAttenuationConfig, TwoBandFilterConfig, ThreeBandFilterConfig, TenBandFilterConfig>;

} // namespace sfFDN