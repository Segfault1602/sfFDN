#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace sfFDN
{
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

/** @brief Enumeration for parallel gain processing modes. */
enum class ParallelGainsMode : uint8_t
{
    Split,   /** < Process input as a single channel and output to multiple channels */
    Merge,   /** < Process each input channel separately and output to one channel */
    Parallel /** < Process each input channel separately and output to the same number of channels */
};

// STRUCTS

struct ScalarFeedbackMatrixInfo
{
    uint32_t matrix_size;
    ScalarMatrixType type;
    std::optional<std::vector<float>> custom_matrix;

    uint32_t rng_seed{0};
    std::optional<float> arg;
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
    float delay;
    uint32_t max_delay;
    sfFDN::DelayInterpolationType interp_type{sfFDN::DelayInterpolationType::Allpass};
    std::optional<sfFDN::ModulationConfig> lfo_config;
};

} // namespace sfFDN