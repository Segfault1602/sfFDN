#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include "rng.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/matrix_gallery.h"
#include "sffdn/time_varying_feedback_matrix.h"

#include "allocation_counter.h"
#include "time_varying_feedback_matrix_internal.h"

namespace
{

constexpr std::array kOrders = {8U, 16U, 32U};
constexpr std::array kModes = {sfFDN::TimeVaryingMatrixMode::Hadamard, sfFDN::TimeVaryingMatrixMode::RealSchur};
constexpr uint32_t kSampleRate = 48000U;
constexpr uint32_t kBlockSize = 256U;
constexpr float kSampleEpsilon = std::numeric_limits<float>::epsilon();
constexpr uint32_t kRealSchurSeed = 0x5EED1234U;
constexpr std::array kRealSchurSeeds = {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U};
constexpr float kSchurSubDiagonalTolerance = 32.0F * kSampleEpsilon;
constexpr std::array kSamplesInModulationCycle = {0U, 3000U, 6000U, 9000U, 12000U, 15000U, 18000U, 21000U};

const char* ModeName(sfFDN::TimeVaryingMatrixMode mode)
{
    return mode == sfFDN::TimeVaryingMatrixMode::Hadamard ? "Hadamard" : "RealSchur";
}

std::vector<sfFDN::ModulationOptions> MakeModulationConfig(uint32_t order, float amplitude)
{
    std::vector<sfFDN::ModulationOptions> config(order / 2U);
    for (uint32_t rotation = 0; rotation < config.size(); ++rotation)
    {
        config[rotation] = {
            .frequency = static_cast<float>(rotation + 1U) / static_cast<float>(kSampleRate),
            .amplitude = amplitude,
            .initial_phase = static_cast<float>((rotation * 7U) % 16U) / 16.0F,
        };
    }
    return config;
}

void FillRandom(std::span<float> data)
{
    sfFDN::RNG rng(0x9E3779B9U);
    for (float& sample : data)
    {
        sample = rng();
    }
}

float MaxAbsDifference(std::span<const float> first, std::span<const float> second)
{
    float max_difference = 0.0F;
    for (size_t index = 0; index < first.size(); ++index)
    {
        max_difference = std::max(max_difference, std::abs(first[index] - second[index]));
    }
    return max_difference;
}

void RequireSamplesWithinAbs(std::span<const float> actual, std::span<const float> expected, float tolerance)
{
    REQUIRE(actual.size() == expected.size());
    for (size_t index = 0; index < actual.size(); ++index)
    {
        REQUIRE_THAT(actual[index], Catch::Matchers::WithinAbs(expected[index], tolerance));
    }
}

void AdvanceToSample(sfFDN::TimeVaryingFeedbackMatrix& matrix, uint32_t order, uint32_t sample)
{
    if (sample == 0U)
    {
        return;
    }

    std::vector<float> silence(order * sample, 0.0F);
    std::vector<float> discarded_output(silence.size(), 0.0F);
    sfFDN::AudioBuffer input_buffer(sample, order, silence);
    sfFDN::AudioBuffer output_buffer(sample, order, discarded_output);
    matrix.Process(input_buffer, output_buffer);
}

std::vector<float> EffectiveMatrixAtSample(sfFDN::TimeVaryingFeedbackMatrix& matrix, uint32_t order, uint32_t sample)
{
    std::vector<float> effective_matrix(order * order, 0.0F);
    std::vector<float> basis_input(order, 0.0F);
    std::vector<float> basis_output(order, 0.0F);
    sfFDN::AudioBuffer input_buffer(1U, order, basis_input);
    sfFDN::AudioBuffer output_buffer(1U, order, basis_output);

    // A block with its samples arranged as an identity matrix would use a different A(n) for every column. Resetting
    // and advancing before each single-sample basis-vector call instead reconstructs all columns of one A(sample).
    for (uint32_t column = 0; column < order; ++column)
    {
        matrix.Clear();
        AdvanceToSample(matrix, order, sample);
        std::fill(basis_input.begin(), basis_input.end(), 0.0F);
        basis_input[column] = 1.0F;
        matrix.Process(input_buffer, output_buffer);

        for (uint32_t row = 0; row < order; ++row)
        {
            effective_matrix[(row * order) + column] = basis_output[row];
        }
    }
    return effective_matrix;
}

float OrthogonalityError(std::span<const float> matrix, uint32_t order)
{
    float sum_squared_error = 0.0F;
    for (uint32_t first_column = 0; first_column < order; ++first_column)
    {
        for (uint32_t second_column = 0; second_column < order; ++second_column)
        {
            float dot_product = 0.0F;
            for (uint32_t row = 0; row < order; ++row)
            {
                dot_product += matrix[(row * order) + first_column] * matrix[(row * order) + second_column];
            }

            const float expected = first_column == second_column ? 1.0F : 0.0F;
            const float error = dot_product - expected;
            sum_squared_error += error * error;
        }
    }
    return std::sqrt(sum_squared_error);
}

struct RealSchurMetrics
{
    uint32_t rotation_blocks{};
    uint32_t scalar_blocks{};
    float off_block_mass{};
    float residual{};
};

RealSchurMetrics MeasureRealSchur(uint32_t order, uint32_t seed = kRealSchurSeed)
{
    const auto matrix_data = sfFDN::GenerateMatrix(order, sfFDN::ScalarMatrixType::Random, seed);
    Eigen::MatrixXf matrix = Eigen::Map<const Eigen::MatrixXf>(matrix_data.data(), order, order);
    if (matrix.determinant() < 0.0F)
    {
        matrix.col(static_cast<Eigen::Index>(order - 1U)) *= -1.0F;
    }
    const Eigen::RealSchur<Eigen::MatrixXf> schur(matrix);
    REQUIRE(schur.info() == Eigen::Success);

    const Eigen::MatrixXf& schur_form = schur.matrixT();
    std::vector<uint32_t> block_indices(order);
    RealSchurMetrics metrics;
    for (uint32_t index = 0; index < order;)
    {
        const bool is_rotation =
            (index + 1U) < order && std::abs(schur_form(index + 1U, index)) > kSchurSubDiagonalTolerance;
        if (is_rotation)
        {
            block_indices[index] = index;
            block_indices[index + 1U] = index;
            ++metrics.rotation_blocks;
            index += 2U;
        }
        else
        {
            block_indices[index] = index;
            ++metrics.scalar_blocks;
            ++index;
        }
    }

    float off_block_sum_squared = 0.0F;
    for (uint32_t row = 0; row < order; ++row)
    {
        for (uint32_t column = 0; column < order; ++column)
        {
            if (block_indices[row] != block_indices[column])
            {
                const float value = schur_form(row, column);
                off_block_sum_squared += value * value;
            }
        }
    }
    metrics.off_block_mass = std::sqrt(off_block_sum_squared);
    metrics.residual = (matrix - (schur.matrixU() * schur_form * schur.matrixU().transpose())).norm();
    return metrics;
}

sfFDN::TimeVaryingFeedbackMatrix MakeMatrix(uint32_t order, float amplitude, sfFDN::TimeVaryingMatrixMode mode,
                                            uint32_t seed = kRealSchurSeed)
{
    return sfFDN::TimeVaryingFeedbackMatrix({.matrix_size = order,
                                             .mode = mode,
                                             .time_varying_config = MakeModulationConfig(order, amplitude),
                                             .rng_seed = seed});
}

void StaticReferenceProcess(std::span<const float> input, std::span<float> output, uint32_t order, uint32_t block_size,
                            std::span<const float> base_angles)
{
    const float normalization = 1.0F / std::sqrt(static_cast<float>(order));
    std::vector<float> first_hadamard(order);
    std::vector<float> rotated(order);
    for (uint32_t sample = 0; sample < block_size; ++sample)
    {
        for (uint32_t row = 0; row < order; ++row)
        {
            float value = 0.0F;
            for (uint32_t column = 0; column < order; ++column)
            {
                const bool negative = (std::popcount(row & column) % 2) != 0;
                const float hadamard = negative ? -normalization : normalization;
                value += hadamard * input[(column * block_size) + sample];
            }
            first_hadamard[row] = value;
        }

        for (uint32_t rotation = 0; rotation < base_angles.size(); ++rotation)
        {
            const uint32_t first_channel = 2U * rotation;
            const uint32_t second_channel = first_channel + 1U;
            const float sine = std::sin(base_angles[rotation]);
            const float cosine = std::cos(base_angles[rotation]);
            rotated[first_channel] = (cosine * first_hadamard[first_channel]) - (sine * first_hadamard[second_channel]);
            rotated[second_channel] =
                (sine * first_hadamard[first_channel]) + (cosine * first_hadamard[second_channel]);
        }

        for (uint32_t row = 0; row < order; ++row)
        {
            float value = 0.0F;
            for (uint32_t column = 0; column < order; ++column)
            {
                const bool negative = (std::popcount(row & column) % 2) != 0;
                const float hadamard = negative ? -normalization : normalization;
                value += hadamard * rotated[column];
            }
            output[(row * block_size) + sample] = value;
        }
    }
}

} // namespace

TEST_CASE("TimeVaryingFeedbackMatrix remains orthogonal over time")
{
    for (const auto mode : kModes)
    {
        for (const uint32_t order : kOrders)
        {
            auto config = MakeModulationConfig(order, 0.8F);
            for (auto& modulation : config)
            {
                modulation.frequency = 2.0F / static_cast<float>(kSampleRate);
            }
            sfFDN::TimeVaryingFeedbackMatrix matrix(
                {.matrix_size = order, .mode = mode, .time_varying_config = config, .rng_seed = kRealSchurSeed});

            float worst_error = 0.0F;
            for (const uint32_t sample : kSamplesInModulationCycle)
            {
                worst_error =
                    std::max(worst_error, OrthogonalityError(EffectiveMatrixAtSample(matrix, order, sample), order));
            }

            const float tolerance = 10.0F * std::sqrt(static_cast<float>(order)) * kSampleEpsilon;
            INFO("mode=" << ModeName(mode) << " order=" << order << " worst ||A^T A - I||_F=" << std::setprecision(10)
                         << worst_error);
            REQUIRE_THAT(worst_error, Catch::Matchers::WithinAbs(0.0F, tolerance));
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix conserves energy")
{
    constexpr double kEnergyRelativeTolerance = 2.0e-6; // Float32 Hadamard and rotation roundoff over 256 samples.

    for (const auto mode : kModes)
    {
        for (const uint32_t order : kOrders)
        {
            auto matrix = MakeMatrix(order, 0.7F, mode);
            std::vector<float> input(order * kBlockSize);
            std::vector<float> output(input.size(), 0.0F);
            FillRandom(input);
            sfFDN::AudioBuffer input_buffer(kBlockSize, order, input);
            sfFDN::AudioBuffer output_buffer(kBlockSize, order, output);
            matrix.Process(input_buffer, output_buffer);

            double input_energy = 0.0;
            double output_energy = 0.0;
            for (size_t index = 0; index < input.size(); ++index)
            {
                input_energy += static_cast<double>(input[index]) * input[index];
                output_energy += static_cast<double>(output[index]) * output[index];
            }
            const double energy_ratio = output_energy / input_energy;

            INFO("mode=" << ModeName(mode) << " order=" << order << " energy ratio=" << std::setprecision(10)
                         << energy_ratio);
            REQUIRE_THAT(energy_ratio, Catch::Matchers::WithinAbs(1.0, kEnergyRelativeTolerance));
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix zero modulation is static")
{
    constexpr float kReferenceTolerance = 2.0e-5F;

    for (const auto mode : kModes)
    {
        for (const uint32_t order : kOrders)
        {
            auto matrix = MakeMatrix(order, 0.0F, mode);
            std::vector<float> base_angles(order / 2U);
            for (uint32_t rotation = 0; rotation < base_angles.size(); ++rotation)
            {
                base_angles[rotation] = 0.17F * static_cast<float>(rotation + 1U);
            }
            matrix.SetBaseAngles(base_angles);

            std::vector<float> input(order * kBlockSize);
            std::vector<float> first_output(input.size(), 0.0F);
            std::vector<float> second_output(input.size(), 0.0F);
            std::vector<float> expected_output(input.size(), 0.0F);
            FillRandom(input);
            sfFDN::AudioBuffer input_buffer(kBlockSize, order, input);
            sfFDN::AudioBuffer first_output_buffer(kBlockSize, order, first_output);
            sfFDN::AudioBuffer second_output_buffer(kBlockSize, order, second_output);
            matrix.Process(input_buffer, first_output_buffer);
            matrix.Process(input_buffer, second_output_buffer);

            RequireSamplesWithinAbs(first_output, second_output, kReferenceTolerance);
            if (mode == sfFDN::TimeVaryingMatrixMode::Hadamard)
            {
                StaticReferenceProcess(input, expected_output, order, kBlockSize, base_angles);
                RequireSamplesWithinAbs(first_output, expected_output, kReferenceTolerance);

                auto identity_matrix = MakeMatrix(order, 0.0F, mode);
                std::vector<float> identity_output(input.size(), 0.0F);
                sfFDN::AudioBuffer identity_output_buffer(kBlockSize, order, identity_output);
                identity_matrix.Process(input_buffer, identity_output_buffer);
                RequireSamplesWithinAbs(identity_output, input, 4.0F * kSampleEpsilon);
            }
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix modulation changes output")
{
    constexpr float kMinimumSubstantialDifference = 0.01F;

    for (const auto mode : kModes)
    {
        for (const uint32_t order : kOrders)
        {
            auto unmodulated_matrix = MakeMatrix(order, 0.0F, mode);
            auto modulated_matrix = MakeMatrix(order, 0.7F, mode);
            std::vector<float> input(order * kBlockSize);
            std::vector<float> unmodulated_output(input.size(), 0.0F);
            std::vector<float> modulated_output(input.size(), 0.0F);
            FillRandom(input);
            sfFDN::AudioBuffer input_buffer(kBlockSize, order, input);
            sfFDN::AudioBuffer unmodulated_buffer(kBlockSize, order, unmodulated_output);
            sfFDN::AudioBuffer modulated_buffer(kBlockSize, order, modulated_output);
            unmodulated_matrix.Process(input_buffer, unmodulated_buffer);
            modulated_matrix.Process(input_buffer, modulated_buffer);

            const float max_difference = MaxAbsDifference(unmodulated_output, modulated_output);
            INFO("mode=" << ModeName(mode) << " order=" << order
                         << " max modulated difference=" << std::setprecision(10) << max_difference);
            REQUIRE(max_difference > kMinimumSubstantialDifference);
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix Process is allocation-free")
{
    for (const auto mode : kModes)
    {
        for (const uint32_t order : kOrders)
        {
            auto matrix = MakeMatrix(order, 0.7F, mode);
            std::vector<float> input(order * kBlockSize);
            std::vector<float> output(input.size(), 0.0F);
            auto aliased = input;
            FillRandom(input);
            aliased = input;
            sfFDN::AudioBuffer input_buffer(kBlockSize, order, input);
            sfFDN::AudioBuffer output_buffer(kBlockSize, order, output);
            sfFDN::AudioBuffer aliased_buffer(kBlockSize, order, aliased);

            size_t allocations = 0U;
            {
                sfFDNTest::ScopedAllocationCounter allocation_counter;
                matrix.Process(input_buffer, output_buffer);
                matrix.Process(aliased_buffer, aliased_buffer);
                allocations = allocation_counter.Count();
            }

            INFO("mode=" << ModeName(mode) << " order=" << order);
            REQUIRE(allocations == 0U);
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix supports aliased processing")
{
    constexpr float kAliasingTolerance = 2.0e-5F;

    for (const auto mode : kModes)
    {
        for (const uint32_t order : kOrders)
        {
            auto matrix = MakeMatrix(order, 0.7F, mode);
            std::vector<float> input(order * kBlockSize);
            std::vector<float> expected_output(input.size(), 0.0F);
            FillRandom(input);
            auto aliased = input;
            sfFDN::AudioBuffer input_buffer(kBlockSize, order, input);
            sfFDN::AudioBuffer expected_buffer(kBlockSize, order, expected_output);
            sfFDN::AudioBuffer aliased_buffer(kBlockSize, order, aliased);

            matrix.Clear();
            matrix.Process(input_buffer, expected_buffer);
            matrix.Clear();
            matrix.Process(aliased_buffer, aliased_buffer);

            RequireSamplesWithinAbs(aliased, expected_output, kAliasingTolerance);
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix is block-partition invariant")
{
    constexpr uint32_t kOrder = 8U;
    constexpr uint32_t kTotalSamples = 200000U;
    constexpr float kFrequency = 16.0F / static_cast<float>(kSampleRate);
    constexpr float kPartitionTolerance = 0.0F;
    constexpr std::array kPartitionSizes = {1U, 37U, 64U, 100U, 128U, 300U};

    for (const auto mode : kModes)
    {
        auto config = MakeModulationConfig(kOrder, 0.7F);
        for (auto& modulation : config)
        {
            modulation.frequency = kFrequency;
        }
        sfFDN::TimeVaryingFeedbackMatrix matrix(
            {.matrix_size = kOrder, .mode = mode, .time_varying_config = config, .rng_seed = kRealSchurSeed});
        std::vector<float> input(kOrder * kTotalSamples);
        std::vector<float> whole_output(input.size(), 0.0F);
        std::vector<float> partitioned_output(input.size(), 0.0F);
        FillRandom(input);
        sfFDN::AudioBuffer input_buffer(kTotalSamples, kOrder, input);
        sfFDN::AudioBuffer whole_output_buffer(kTotalSamples, kOrder, whole_output);
        sfFDN::AudioBuffer partitioned_output_buffer(kTotalSamples, kOrder, partitioned_output);

        matrix.Clear();
        matrix.Process(input_buffer, whole_output_buffer);
        for (const uint32_t partition_size : kPartitionSizes)
        {
            matrix.Clear();
            for (uint32_t offset = 0; offset < kTotalSamples; offset += partition_size)
            {
                const uint32_t size = std::min(partition_size, kTotalSamples - offset);
                const auto input_partition = input_buffer.Offset(offset, size);
                auto output_partition = partitioned_output_buffer.Offset(offset, size);
                matrix.Process(input_partition, output_partition);
            }

            const float partitioned_residual = MaxAbsDifference(whole_output, partitioned_output);
            INFO("mode=" << ModeName(mode) << " partition size=" << partition_size
                         << " residual=" << std::setprecision(10) << partitioned_residual);
            REQUIRE_THAT(partitioned_residual, Catch::Matchers::WithinAbs(0.0F, kPartitionTolerance));
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix Clear resets modulation phase")
{
    constexpr float kClearTolerance = 2.0e-5F;

    for (const auto mode : kModes)
    {
        for (const uint32_t order : kOrders)
        {
            auto matrix = MakeMatrix(order, 0.7F, mode);
            std::vector<float> input(order * kBlockSize);
            std::vector<float> first_output(input.size(), 0.0F);
            std::vector<float> second_output(input.size(), 0.0F);
            FillRandom(input);
            sfFDN::AudioBuffer input_buffer(kBlockSize, order, input);
            sfFDN::AudioBuffer first_output_buffer(kBlockSize, order, first_output);
            sfFDN::AudioBuffer second_output_buffer(kBlockSize, order, second_output);

            matrix.Process(input_buffer, first_output_buffer);
            matrix.Clear();
            matrix.Process(input_buffer, second_output_buffer);

            RequireSamplesWithinAbs(second_output, first_output, kClearTolerance);
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix Clone continues modulation phase")
{
    constexpr float kCloneTolerance = 2.0e-5F;

    for (const auto mode : kModes)
    {
        for (const uint32_t order : kOrders)
        {
            auto matrix = MakeMatrix(order, 0.7F, mode);
            std::vector<float> input(order * kBlockSize);
            std::vector<float> discarded_output(input.size(), 0.0F);
            std::vector<float> original_output(input.size(), 0.0F);
            std::vector<float> clone_output(input.size(), 0.0F);
            FillRandom(input);
            sfFDN::AudioBuffer input_buffer(kBlockSize, order, input);
            sfFDN::AudioBuffer discarded_output_buffer(kBlockSize, order, discarded_output);
            sfFDN::AudioBuffer original_output_buffer(kBlockSize, order, original_output);
            sfFDN::AudioBuffer clone_output_buffer(kBlockSize, order, clone_output);

            matrix.Process(input_buffer, discarded_output_buffer);
            auto clone = matrix.Clone();
            matrix.Process(input_buffer, original_output_buffer);
            clone->Process(input_buffer, clone_output_buffer);

            RequireSamplesWithinAbs(clone_output, original_output, kCloneTolerance);
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix constructor validates options")
{
    for (const uint32_t order : kOrders)
    {
        REQUIRE_THROWS_AS(sfFDN::TimeVaryingFeedbackMatrix({.matrix_size = order + 1U, .time_varying_config = {}}),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(sfFDN::TimeVaryingFeedbackMatrix({.matrix_size = 12U, .time_varying_config = {}}),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(sfFDN::TimeVaryingFeedbackMatrix({.matrix_size = 0U, .time_varying_config = {}}),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(sfFDN::TimeVaryingFeedbackMatrix({.matrix_size = order + 1U,
                                                            .mode = sfFDN::TimeVaryingMatrixMode::RealSchur,
                                                            .time_varying_config = {}}),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(
            sfFDN::TimeVaryingFeedbackMatrix(
                {.matrix_size = order, .time_varying_config = std::vector<sfFDN::ModulationOptions>(order / 2U - 1U)}),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            sfFDN::TimeVaryingFeedbackMatrix(
                {.matrix_size = order,
                 .time_varying_config = std::vector<sfFDN::ModulationOptions>(order / 2U, {.amplitude = 1.01F})}),
            std::invalid_argument);
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix rejects invalid modulation parameters")
{
    constexpr uint32_t kOrder = 8U;
    auto config = MakeModulationConfig(kOrder, 0.7F);

    config[0].amplitude = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        sfFDN::TimeVaryingFeedbackMatrix(
            {.matrix_size = kOrder, .mode = sfFDN::TimeVaryingMatrixMode::Hadamard, .time_varying_config = config}),
        std::invalid_argument);

    config = MakeModulationConfig(kOrder, 0.7F);
    config[0].frequency = std::numeric_limits<float>::infinity();
    REQUIRE_THROWS_AS(
        sfFDN::TimeVaryingFeedbackMatrix(
            {.matrix_size = kOrder, .mode = sfFDN::TimeVaryingMatrixMode::Hadamard, .time_varying_config = config}),
        std::invalid_argument);

    config = MakeModulationConfig(kOrder, 0.7F);
    config[0].initial_phase = -0.001F;
    REQUIRE_THROWS_AS(
        sfFDN::TimeVaryingFeedbackMatrix(
            {.matrix_size = kOrder, .mode = sfFDN::TimeVaryingMatrixMode::Hadamard, .time_varying_config = config}),
        std::invalid_argument);

    auto matrix = MakeMatrix(kOrder, 0.7F, sfFDN::TimeVaryingMatrixMode::Hadamard);
    std::array<float, kOrder / 2U> values{};

    values.fill(0.001F);
    values[0] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(matrix.SetLfoFrequency(values), std::invalid_argument);
    REQUIRE_THROWS_AS(matrix.SetLfoAmplitude(values), std::invalid_argument);
    REQUIRE_THROWS_AS(matrix.SetLfoPhaseOffset(values), std::invalid_argument);
    REQUIRE_THROWS_AS(matrix.SetBaseAngles(values), std::invalid_argument);

    config = MakeModulationConfig(kOrder, 0.7F);
    config[0].initial_phase = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(matrix.SetModulation(config), std::invalid_argument);

    config = MakeModulationConfig(kOrder, 0.7F);
    config[0].initial_phase = 1.001F;
    REQUIRE_THROWS_AS(matrix.SetModulation(config), std::invalid_argument);

    values.fill(0.001F);
    values[0] = -0.001F;
    REQUIRE_THROWS_AS(matrix.SetLfoPhaseOffset(values), std::invalid_argument);

    values[0] = std::numeric_limits<float>::infinity();
    REQUIRE_THROWS_AS(matrix.SetBaseAngles(values), std::invalid_argument);
}

TEST_CASE("TimeVaryingFeedbackMatrix range-reduces large base angles")
{
    constexpr uint32_t kOrder = 8U;
    constexpr float kReferenceTolerance = 2.0e-5F;
    const std::array<float, kOrder / 2U> base_angles = {1.0e6F, -1.0e8F, 1.0e7F, -1.0e6F};
    std::vector<float> reduced_angles(base_angles.size());
    for (size_t index = 0; index < base_angles.size(); ++index)
    {
        reduced_angles[index] = static_cast<float>(
            std::remainder(static_cast<double>(base_angles[index]), 2.0 * std::numbers::pi_v<double>));
    }

    auto matrix = MakeMatrix(kOrder, 0.0F, sfFDN::TimeVaryingMatrixMode::Hadamard);
    matrix.SetBaseAngles(base_angles);
    std::vector<float> input(kOrder * kBlockSize);
    std::vector<float> output(input.size(), 0.0F);
    std::vector<float> expected(output.size(), 0.0F);
    FillRandom(input);
    const sfFDN::AudioBuffer input_buffer(kBlockSize, kOrder, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kOrder, output);
    matrix.Process(input_buffer, output_buffer);
    StaticReferenceProcess(input, expected, kOrder, kBlockSize, reduced_angles);

    RequireSamplesWithinAbs(output, expected, kReferenceTolerance);
}

TEST_CASE("TimeVaryingFeedbackMatrix RealSchur supports all even orders")
{
    for (const uint32_t order : {6U, 8U, 10U, 12U, 16U, 32U})
    {
        const auto metrics = MeasureRealSchur(order);
        auto matrix = MakeMatrix(order, 0.7F, sfFDN::TimeVaryingMatrixMode::RealSchur);
        std::vector<float> input(order * kBlockSize);
        std::vector<float> output(input.size(), 0.0F);
        FillRandom(input);
        sfFDN::AudioBuffer input_buffer(kBlockSize, order, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, order, output);
        matrix.Process(input_buffer, output_buffer);
        float orthogonality_error = 0.0F;
        for (const uint32_t sample : kSamplesInModulationCycle)
        {
            orthogonality_error = std::max(orthogonality_error,
                                           OrthogonalityError(EffectiveMatrixAtSample(matrix, order, sample), order));
        }
        const float orthogonality_tolerance = 10.0F * std::sqrt(static_cast<float>(order)) * kSampleEpsilon;

        INFO("order=" << order << " rotation blocks=" << metrics.rotation_blocks << " scalar blocks="
                      << metrics.scalar_blocks << " off-block mass=" << std::setprecision(10) << metrics.off_block_mass
                      << " Schur residual=" << metrics.residual << " worst ||A^T A - I||_F=" << orthogonality_error);
        REQUIRE(metrics.rotation_blocks == order / 2U);
        REQUIRE(metrics.scalar_blocks == 0U);
        REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
        REQUIRE_THAT(orthogonality_error, Catch::Matchers::WithinAbs(0.0F, orthogonality_tolerance));
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix RealSchur rotation blocks are seed-independent")
{
    for (const uint32_t order : {6U, 8U, 10U, 12U, 16U, 32U})
    {
        for (const uint32_t seed : kRealSchurSeeds)
        {
            const auto metrics = MeasureRealSchur(order, seed);
            INFO("order=" << order << " seed=" << seed << " rotation blocks=" << metrics.rotation_blocks
                          << " scalar blocks=" << metrics.scalar_blocks);
            REQUIRE(metrics.rotation_blocks == order / 2U);
            REQUIRE(metrics.scalar_blocks == 0U);
            REQUIRE_NOTHROW(MakeMatrix(order, 0.7F, sfFDN::TimeVaryingMatrixMode::RealSchur, seed));
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix RealSchur is deterministic with the default seed")
{
    for (const uint32_t order : {6U, 8U, 10U, 12U, 16U})
    {
        const sfFDN::TimeVaryingFeedbackMatrixOptions options = {
            .matrix_size = order,
            .mode = sfFDN::TimeVaryingMatrixMode::RealSchur,
            .time_varying_config = MakeModulationConfig(order, 0.7F),
        };
        sfFDN::TimeVaryingFeedbackMatrix first(options);
        sfFDN::TimeVaryingFeedbackMatrix second(options);
        std::vector<float> input(order * kBlockSize);
        std::vector<float> first_output(input.size(), 0.0F);
        std::vector<float> second_output(input.size(), 0.0F);
        FillRandom(input);
        sfFDN::AudioBuffer input_buffer(kBlockSize, order, input);
        sfFDN::AudioBuffer first_output_buffer(kBlockSize, order, first_output);
        sfFDN::AudioBuffer second_output_buffer(kBlockSize, order, second_output);
        first.Process(input_buffer, first_output_buffer);
        second.Process(input_buffer, second_output_buffer);

        INFO("order=" << order);
        REQUIRE(first_output == second_output);
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix RealSchur supports scalar Schur blocks")
{
    constexpr uint32_t kOrder = 6U;
    constexpr float kAngleA = 0.7F;
    constexpr float kAngleB = 1.3F;
    constexpr float kMatrixTolerance = 2.0e-5F;
    constexpr double kEnergyRelativeTolerance = 2.0e-6;
    constexpr float kMinimumSubstantialDifference = 0.01F;

    Eigen::MatrixXf custom_basis = Eigen::MatrixXf::Zero(kOrder, kOrder);
    const auto set_rotation = [&custom_basis](uint32_t start, float angle) {
        const float sine = std::sin(angle);
        const float cosine = std::cos(angle);
        custom_basis(start, start) = cosine;
        custom_basis(start, start + 1U) = -sine;
        custom_basis(start + 1U, start) = sine;
        custom_basis(start + 1U, start + 1U) = cosine;
    };
    set_rotation(0U, kAngleA);
    custom_basis(2U, 2U) = 1.0F;
    custom_basis(3U, 3U) = -1.0F;
    set_rotation(4U, kAngleB);
    std::vector<float> custom_basis_data;
    custom_basis_data.reserve(static_cast<size_t>(custom_basis.size()));
    for (uint32_t column = 0; column < kOrder; ++column)
    {
        for (uint32_t row = 0; row < kOrder; ++row)
        {
            custom_basis_data.push_back(custom_basis(row, column));
        }
    }

    const std::vector<sfFDN::ModulationOptions> unmodulated_config = {
        {.frequency = 1.0F / static_cast<float>(kSampleRate), .amplitude = 0.0F, .initial_phase = 0.125F},
        {.frequency = 2.0F / static_cast<float>(kSampleRate), .amplitude = 0.0F, .initial_phase = 0.75F},
    };
    const sfFDN::TimeVaryingFeedbackMatrixOptions unmodulated_options = {
        .matrix_size = kOrder,
        .mode = sfFDN::TimeVaryingMatrixMode::RealSchur,
        .time_varying_config = unmodulated_config,
    };
    auto unmodulated_matrix = sfFDN::detail::TimeVaryingFeedbackMatrixTestAccess::Create(
        unmodulated_options, std::span<const float>(custom_basis_data));

    REQUIRE(unmodulated_matrix.RotationBlockCount() == 2U);
    REQUIRE(kOrder - (2U * unmodulated_matrix.RotationBlockCount()) == 2U);

    std::vector<float> expected_matrix(kOrder * kOrder);
    for (uint32_t row = 0; row < kOrder; ++row)
    {
        for (uint32_t column = 0; column < kOrder; ++column)
        {
            expected_matrix[(row * kOrder) + column] = custom_basis(row, column);
        }
    }
    const auto effective_matrix = EffectiveMatrixAtSample(unmodulated_matrix, kOrder, 0U);
    RequireSamplesWithinAbs(effective_matrix, expected_matrix, kMatrixTolerance);
    REQUIRE_THAT(OrthogonalityError(effective_matrix, kOrder), Catch::Matchers::WithinAbs(0.0F, kMatrixTolerance));

    auto modulated_config = unmodulated_config;
    for (auto& modulation : modulated_config)
    {
        modulation.amplitude = 0.7F;
    }
    auto modulated_options = unmodulated_options;
    modulated_options.time_varying_config = modulated_config;
    auto modulated_matrix = sfFDN::detail::TimeVaryingFeedbackMatrixTestAccess::Create(
        modulated_options, std::span<const float>(custom_basis_data));

    std::vector<float> input(kOrder * kBlockSize);
    std::vector<float> unmodulated_output(input.size(), 0.0F);
    std::vector<float> modulated_output(input.size(), 0.0F);
    FillRandom(input);
    const sfFDN::AudioBuffer input_buffer(kBlockSize, kOrder, input);
    sfFDN::AudioBuffer unmodulated_output_buffer(kBlockSize, kOrder, unmodulated_output);
    sfFDN::AudioBuffer modulated_output_buffer(kBlockSize, kOrder, modulated_output);
    unmodulated_matrix.Process(input_buffer, unmodulated_output_buffer);
    modulated_matrix.Process(input_buffer, modulated_output_buffer);

    double input_energy = 0.0;
    double output_energy = 0.0;
    for (size_t index = 0; index < input.size(); ++index)
    {
        input_energy += static_cast<double>(input[index]) * input[index];
        output_energy += static_cast<double>(modulated_output[index]) * modulated_output[index];
    }
    REQUIRE(std::ranges::all_of(modulated_output, [](float sample) { return std::isfinite(sample); }));
    REQUIRE_THAT(output_energy / input_energy, Catch::Matchers::WithinAbs(1.0, kEnergyRelativeTolerance));
    REQUIRE(MaxAbsDifference(unmodulated_output, modulated_output) > kMinimumSubstantialDifference);
}

TEST_CASE("TimeVaryingFeedbackMatrix GetMatrix matches the matrix Process applies")
{
    // GetMatrix evaluates each LFO phase in closed form, while Process accumulates it one increment per sample. The
    // two therefore drift apart by float32 rounding that grows with the sample index, so the tolerance tracks it.
    constexpr std::array kProbeSamples = {0U, 1U, 64U, 1024U, 4096U};
    constexpr float kPerSampleAngleDrift = 4.0F * kSampleEpsilon;

    for (const auto mode : kModes)
    {
        for (const uint32_t order : kOrders)
        {
            const auto config = MakeModulationConfig(order, 0.8F);
            sfFDN::TimeVaryingFeedbackMatrix matrix(
                {.matrix_size = order, .mode = mode, .time_varying_config = config, .rng_seed = kRealSchurSeed});

            std::vector<float> materialized(static_cast<size_t>(order) * order, 0.0F);
            for (const uint32_t sample : kProbeSamples)
            {
                REQUIRE(matrix.GetMatrix(materialized, sample));
                const auto processed = EffectiveMatrixAtSample(matrix, order, sample);

                // GetMatrix writes column-major, the test helper row-major, hence the transposed index.
                float worst_difference = 0.0F;
                for (uint32_t row = 0; row < order; ++row)
                {
                    for (uint32_t column = 0; column < order; ++column)
                    {
                        const float difference =
                            std::abs(materialized[(column * order) + row] - processed[(row * order) + column]);
                        worst_difference = std::max(worst_difference, difference);
                    }
                }

                const float tolerance =
                    std::sqrt(static_cast<float>(order)) * (16.0F * kSampleEpsilon + (sample * kPerSampleAngleDrift));
                INFO("mode=" << ModeName(mode) << " order=" << order << " sample=" << sample
                             << " worst difference=" << std::setprecision(10) << worst_difference);
                REQUIRE_THAT(worst_difference, Catch::Matchers::WithinAbs(0.0F, tolerance));
            }
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix GetMatrix stays orthogonal and rejects bad spans")
{
    for (const auto mode : kModes)
    {
        for (const uint32_t order : kOrders)
        {
            sfFDN::TimeVaryingFeedbackMatrix matrix({.matrix_size = order,
                                                     .mode = mode,
                                                     .time_varying_config = MakeModulationConfig(order, 0.8F),
                                                     .rng_seed = kRealSchurSeed});

            const size_t element_count = static_cast<size_t>(order) * order;
            std::vector<float> materialized(element_count, 0.0F);
            for (const uint32_t sample : kSamplesInModulationCycle)
            {
                REQUIRE(matrix.GetMatrix(materialized, sample));
                // Orthogonality is transpose-invariant, so the helper's row-major reading is valid here.
                const float tolerance = 10.0F * std::sqrt(static_cast<float>(order)) * kSampleEpsilon;
                INFO("mode=" << ModeName(mode) << " order=" << order << " sample=" << sample);
                REQUIRE_THAT(OrthogonalityError(materialized, order), Catch::Matchers::WithinAbs(0.0F, tolerance));
            }

            std::vector<float> too_small(element_count - 1U, 0.0F);
            std::vector<float> too_large(element_count + 1U, 0.0F);
            REQUIRE_FALSE(matrix.GetMatrix(too_small));
            REQUIRE_FALSE(matrix.GetMatrix(too_large));
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrix GetMatrix does not disturb processing")
{
    constexpr uint32_t kOrder = 8U;
    constexpr uint32_t kBlockCount = 4U;

    for (const auto mode : kModes)
    {
        const sfFDN::TimeVaryingFeedbackMatrixOptions options{.matrix_size = kOrder,
                                                              .mode = mode,
                                                              .time_varying_config = MakeModulationConfig(kOrder, 0.7F),
                                                              .rng_seed = kRealSchurSeed};
        sfFDN::TimeVaryingFeedbackMatrix reference(options);
        sfFDN::TimeVaryingFeedbackMatrix probed(options);

        std::vector<float> input(static_cast<size_t>(kOrder) * kBlockSize);
        FillRandom(input);
        const sfFDN::AudioBuffer input_buffer(kBlockSize, kOrder, input);

        std::vector<float> reference_output(input.size(), 0.0F);
        std::vector<float> probed_output(input.size(), 0.0F);
        sfFDN::AudioBuffer reference_buffer(kBlockSize, kOrder, reference_output);
        sfFDN::AudioBuffer probed_buffer(kBlockSize, kOrder, probed_output);
        std::vector<float> materialized(static_cast<size_t>(kOrder) * kOrder, 0.0F);

        for (uint32_t block = 0; block < kBlockCount; ++block)
        {
            reference.Process(input_buffer, reference_buffer);
            REQUIRE(probed.GetMatrix(materialized, block * kBlockSize));
            probed.Process(input_buffer, probed_buffer);
            REQUIRE(probed.GetMatrix(materialized, block * kBlockSize));

            INFO("mode=" << ModeName(mode) << " block=" << block);
            REQUIRE(reference_output == probed_output);
        }
    }
}
