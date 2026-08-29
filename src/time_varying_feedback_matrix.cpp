// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "sffdn/time_varying_feedback_matrix.h"

#include "matrix_multiplication.h"
#include "sffdn/matrix_gallery.h"
#include "sincos.h"
#include "time_varying_feedback_matrix_internal.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr size_t kChunkSize = 128U;
constexpr float kSchurSubDiagonalTolerance = 32.0F * std::numeric_limits<float>::epsilon();
// Both norms accumulate O(order) float32 round-off from Eigen's real Schur iteration.
constexpr float kSchurErrorTolerancePerOrder = 256.0F * std::numeric_limits<float>::epsilon();
constexpr uint32_t kDefaultRealSchurSeed = 0x5EED1234U;

void ValidateModulationOption(const sfFDN::ModulationOptions& modulation)
{
    if (!std::isfinite(modulation.frequency))
    {
        throw std::invalid_argument("TimeVaryingFeedbackMatrix: LFO frequency must be finite");
    }

    if (!(std::abs(modulation.amplitude) <= 1.0F))
    {
        throw std::invalid_argument("TimeVaryingFeedbackMatrix: LFO amplitude must be in [-1, 1]");
    }

    if (!std::isfinite(modulation.initial_phase) || modulation.initial_phase < 0.0F || modulation.initial_phase > 1.0F)
    {
        throw std::invalid_argument("TimeVaryingFeedbackMatrix: LFO initial phase must be in [0, 1]");
    }
}

uint32_t ValidateOptions(const sfFDN::TimeVaryingFeedbackMatrixOptions& options)
{
    if (options.mode != sfFDN::TimeVaryingMatrixMode::Hadamard &&
        options.mode != sfFDN::TimeVaryingMatrixMode::RealSchur)
    {
        throw std::invalid_argument("TimeVaryingFeedbackMatrix: unknown matrix mode");
    }

    if (options.matrix_size < 2U || (options.matrix_size % 2U) != 0U)
    {
        throw std::invalid_argument("TimeVaryingFeedbackMatrix: matrix_size must be even and at least two");
    }

    if (options.mode == sfFDN::TimeVaryingMatrixMode::Hadamard && !std::has_single_bit(options.matrix_size))
    {
        throw std::invalid_argument(
            "TimeVaryingFeedbackMatrix: Hadamard mode requires an even power-of-two matrix_size");
    }

    for (const auto& modulation : options.time_varying_config)
    {
        ValidateModulationOption(modulation);
    }

    return options.matrix_size;
}

void ValidateRotationCount(size_t actual_count, size_t expected_count, const char* parameter_name)
{
    if (actual_count != expected_count)
    {
        throw std::invalid_argument(std::string(parameter_name) + " (got " + std::to_string(actual_count) +
                                    ", expected " + std::to_string(expected_count) + ")");
    }
}

// This writes a separate destination buffer, so it is safe when the caller's original input and output alias.
// Matrix storage is column-major: matrix[column * order + row].
void DenseMatVecBlock(std::span<const float> matrix, bool transpose, const sfFDN::AudioBuffer& input,
                      sfFDN::AudioBuffer& output, uint32_t order, size_t block_size) noexcept SFFDN_NONBLOCKING
{
    for (uint32_t output_channel = 0; output_channel < order; ++output_channel)
    {
        const auto output_channel_data = output.GetChannelSpan(output_channel).first(block_size);
        for (uint32_t input_channel = 0; input_channel < order; ++input_channel)
        {
            const size_t matrix_index =
                transpose ? (output_channel * order) + input_channel : (input_channel * order) + output_channel;
            const float coefficient = matrix[matrix_index];
            const auto input_channel_data = input.GetChannelSpan(input_channel).first(block_size);
            if (input_channel == 0U)
            {
                for (size_t sample = 0; sample < block_size; ++sample)
                {
                    output_channel_data[sample] = coefficient * input_channel_data[sample];
                }
            }
            else
            {
                for (size_t sample = 0; sample < block_size; ++sample)
                {
                    output_channel_data[sample] += coefficient * input_channel_data[sample];
                }
            }
        }
    }
}

// Applies one fixed 2x2 rotation per block, holding each angle constant across the whole buffer. Process uses
// per-sample angles instead; this variant exists so GetMatrix can materialize a single snapshot of the matrix.
void ApplyFixedRotationsBlock(sfFDN::AudioBuffer& buffer, std::span<const uint32_t> rotation_starts,
                              std::span<const float> angles, size_t block_size)
{
    for (size_t rotation = 0; rotation < rotation_starts.size(); ++rotation)
    {
        float sine = 0.0F;
        float cosine = 0.0F;
        sfFDN::SinCosUnit(angles[rotation], sine, cosine);

        const uint32_t first_channel = rotation_starts[rotation];
        auto first_channel_data = buffer.GetChannelSpan(first_channel).first(block_size);
        auto second_channel_data = buffer.GetChannelSpan(first_channel + 1U).first(block_size);
        for (size_t sample = 0; sample < block_size; ++sample)
        {
            const float first = first_channel_data[sample];
            const float second = second_channel_data[sample];
            first_channel_data[sample] = (cosine * first) - (sine * second);
            second_channel_data[sample] = (sine * first) + (cosine * second);
        }
    }
}

} // namespace

namespace sfFDN
{

TimeVaryingFeedbackMatrix detail::TimeVaryingFeedbackMatrixTestAccess::Create(
    const TimeVaryingFeedbackMatrixOptions& options, std::span<const float> custom_base_matrix)
{
    return {options, custom_base_matrix};
}

TimeVaryingFeedbackMatrix::TimeVaryingFeedbackMatrix(const TimeVaryingFeedbackMatrixOptions& options)
    : TimeVaryingFeedbackMatrix(options, {})
{
}

TimeVaryingFeedbackMatrix::TimeVaryingFeedbackMatrix(const TimeVaryingFeedbackMatrixOptions& options,
                                                     std::span<const float> custom_base_matrix)
    : order_(ValidateOptions(options))
    , mode_(options.mode)
    , scalar_signs_(order_, 1.0F)
    , scratch_(order_ * kChunkSize)
{
    if (mode_ == TimeVaryingMatrixMode::Hadamard)
    {
        base_angles_.assign(order_ / 2U, 0.0F);
        rotation_starts_.resize(order_ / 2U);
        for (uint32_t rotation = 0; rotation < rotation_starts_.size(); ++rotation)
        {
            rotation_starts_[rotation] = 2U * rotation;
        }
    }
    else
    {
        Eigen::MatrixXf base_matrix;
        if (custom_base_matrix.empty())
        {
            // Mapped the same way ScalarFeedbackMatrix consumes matrix data: the library's flat matrix layout is
            // column-major, which is exactly what Eigen's default Map expects. Any orthogonal basis is valid here,
            // so the Schur decomposition below is unaffected by orientation either way.
            const uint32_t seed = options.rng_seed == 0U ? kDefaultRealSchurSeed : options.rng_seed;
            const auto matrix_data = GenerateMatrix(order_, ScalarMatrixType::Random, seed);
            base_matrix = Eigen::Map<const Eigen::MatrixXf>(matrix_data.data(), static_cast<Eigen::Index>(order_),
                                                            static_cast<Eigen::Index>(order_));
        }
        else
        {
            if (custom_base_matrix.size() != static_cast<size_t>(order_) * order_)
            {
                throw std::invalid_argument(
                    "TimeVaryingFeedbackMatrix: custom RealSchur basis must contain matrix_size squared values");
            }
            base_matrix = Eigen::Map<const Eigen::MatrixXf>(
                custom_base_matrix.data(), static_cast<Eigen::Index>(order_), static_cast<Eigen::Index>(order_));
        }

        // RandomOrthogonal is Haar-distributed on O(N), not SO(N). In an even-order matrix, a negative determinant
        // forces real eigenvalues and therefore fewer modulatable 2x2 rotation blocks. Flip one column to obtain a
        // proper rotation. Do not remove this: it ensures every channel can participate in a modulatable Schur block.
        if (custom_base_matrix.empty() && base_matrix.determinant() < 0.0F)
        {
            base_matrix.col(static_cast<Eigen::Index>(order_ - 1U)) *= -1.0F;
        }
        const Eigen::RealSchur<Eigen::MatrixXf> schur(base_matrix);
        if (schur.info() != Eigen::Success)
        {
            throw std::runtime_error("TimeVaryingFeedbackMatrix: RealSchur decomposition failed");
        }

        const Eigen::MatrixXf& schur_basis = schur.matrixU();
        const Eigen::MatrixXf& schur_form = schur.matrixT();
        std::vector<uint32_t> block_indices(order_);
        for (uint32_t index = 0; index < order_;)
        {
            const auto eigen_index = static_cast<Eigen::Index>(index);
            const bool is_rotation = (index + 1U) < order_ &&
                                     std::abs(schur_form(eigen_index + 1, eigen_index)) > kSchurSubDiagonalTolerance;
            if (is_rotation)
            {
                block_indices[index] = index;
                block_indices[index + 1U] = index;
                rotation_starts_.push_back(index);
                base_angles_.push_back(
                    std::atan2(schur_form(eigen_index + 1, eigen_index), schur_form(eigen_index, eigen_index)));
                index += 2U;
            }
            else
            {
                block_indices[index] = index;
                scalar_signs_[index] =
                    schur_form(static_cast<Eigen::Index>(index), static_cast<Eigen::Index>(index)) >= 0.0F ? 1.0F
                                                                                                           : -1.0F;
                ++index;
            }
        }

        float off_block_sum_squared = 0.0F;
        for (uint32_t row = 0; row < order_; ++row)
        {
            for (uint32_t column = 0; column < order_; ++column)
            {
                if (block_indices[row] != block_indices[column])
                {
                    const float value = schur_form(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
                    off_block_sum_squared += value * value;
                }
            }
        }
        const float off_block_mass = std::sqrt(off_block_sum_squared);
        const float error_tolerance = kSchurErrorTolerancePerOrder * std::sqrt(static_cast<float>(order_));
        if (off_block_mass > error_tolerance)
        {
            throw std::runtime_error("TimeVaryingFeedbackMatrix: RealSchur off-block mass " +
                                     std::to_string(off_block_mass) + " exceeds tolerance " +
                                     std::to_string(error_tolerance));
        }

        const float residual = (base_matrix - (schur_basis * schur_form * schur_basis.transpose())).norm();
        if (residual > error_tolerance)
        {
            throw std::runtime_error("TimeVaryingFeedbackMatrix: RealSchur residual " + std::to_string(residual) +
                                     " exceeds tolerance " + std::to_string(error_tolerance));
        }

        // Reorthogonalize Eigen's float32 output before the repeated runtime transforms.
        const Eigen::HouseholderQR<Eigen::MatrixXf> basis_qr(schur_basis);
        const Eigen::MatrixXf runtime_basis = basis_qr.householderQ();
        for (size_t rotation = 0; rotation < rotation_starts_.size(); ++rotation)
        {
            const auto first_channel = static_cast<Eigen::Index>(rotation_starts_[rotation]);
            const Eigen::Index second_channel = first_channel + 1;
            const float basis_orientation = runtime_basis.col(first_channel).dot(schur_basis.col(first_channel)) *
                                            runtime_basis.col(second_channel).dot(schur_basis.col(second_channel));
            if (basis_orientation < 0.0F)
            {
                base_angles_[rotation] = -base_angles_[rotation];
            }
        }
        schur_basis_.resize(static_cast<size_t>(runtime_basis.size()));
        for (uint32_t column = 0; column < order_; ++column)
        {
            for (uint32_t row = 0; row < order_; ++row)
            {
                schur_basis_[(column * order_) + row] =
                    runtime_basis(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
            }
        }
    }

    lfos_.resize(base_angles_.size());
    lfo_phases_.resize(base_angles_.size(), 0.0F);
    for (auto& lfo : lfos_)
    {
        lfo.SetAmplitude(0.0F);
    }

    if (!options.time_varying_config.empty())
    {
        SetModulation(options.time_varying_config);
    }
}

void TimeVaryingFeedbackMatrix::SetModulation(std::span<const ModulationOptions> modulation_configs)
{
    if (modulation_configs.empty())
    {
        for (auto& lfo : lfos_)
        {
            lfo.SetFrequency(0.0F);
            lfo.SetAmplitude(0.0F);
            lfo.SetPhaseOffset(0.0F);
        }
        return;
    }

    ValidateRotationCount(modulation_configs.size(), lfos_.size(),
                          "TimeVaryingFeedbackMatrix: expected one modulation option per rotation block");
    for (const auto& modulation : modulation_configs)
    {
        ValidateModulationOption(modulation);
    }

    for (size_t index = 0; index < lfos_.size(); ++index)
    {
        lfos_[index].SetFrequency(modulation_configs[index].frequency);
        lfos_[index].SetAmplitude(modulation_configs[index].amplitude);
        lfos_[index].SetPhaseOffset(modulation_configs[index].initial_phase);
    }
}

void TimeVaryingFeedbackMatrix::SetLfoFrequency(std::span<const float> frequencies)
{
    ValidateRotationCount(frequencies.size(), lfos_.size(),
                          "TimeVaryingFeedbackMatrix: expected one frequency per rotation block");
    for (const float frequency : frequencies)
    {
        if (!std::isfinite(frequency))
        {
            throw std::invalid_argument("TimeVaryingFeedbackMatrix: LFO frequency must be finite");
        }
    }

    for (size_t index = 0; index < lfos_.size(); ++index)
    {
        lfos_[index].SetFrequency(frequencies[index]);
    }
}

void TimeVaryingFeedbackMatrix::SetLfoAmplitude(std::span<const float> amplitudes)
{
    ValidateRotationCount(amplitudes.size(), lfos_.size(),
                          "TimeVaryingFeedbackMatrix: expected one amplitude per rotation block");
    for (const float amplitude : amplitudes)
    {
        if (!(std::abs(amplitude) <= 1.0F))
        {
            throw std::invalid_argument("TimeVaryingFeedbackMatrix: LFO amplitude must be in [-1, 1]");
        }
    }

    for (size_t index = 0; index < lfos_.size(); ++index)
    {
        lfos_[index].SetAmplitude(amplitudes[index]);
    }
}

void TimeVaryingFeedbackMatrix::SetLfoPhaseOffset(std::span<const float> phase_offsets)
{
    ValidateRotationCount(phase_offsets.size(), lfos_.size(),
                          "TimeVaryingFeedbackMatrix: expected one phase offset per rotation block");
    for (const float phase_offset : phase_offsets)
    {
        if (!std::isfinite(phase_offset) || phase_offset < 0.0F || phase_offset > 1.0F)
        {
            throw std::invalid_argument("TimeVaryingFeedbackMatrix: LFO phase offset must be in [0, 1]");
        }
    }

    for (size_t index = 0; index < lfos_.size(); ++index)
    {
        lfos_[index].SetPhaseOffset(phase_offsets[index]);
    }
}

void TimeVaryingFeedbackMatrix::SetBaseAngles(std::span<const float> radians)
{
    ValidateRotationCount(radians.size(), base_angles_.size(),
                          "TimeVaryingFeedbackMatrix: expected one base angle per rotation block");
    constexpr double kTwoPi = 2.0 * std::numbers::pi_v<double>;
    for (size_t index = 0; index < radians.size(); ++index)
    {
        if (!std::isfinite(radians[index]))
        {
            throw std::invalid_argument("TimeVaryingFeedbackMatrix: base angles must be finite");
        }
        base_angles_[index] = static_cast<float>(std::remainder(static_cast<double>(radians[index]), kTwoPi));
    }
}

void TimeVaryingFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.ChannelCount() == order_);
    assert(output.ChannelCount() == order_);
    assert(input.SampleCount() == output.SampleCount());

    if (mode_ == TimeVaryingMatrixMode::Hadamard)
    {
        HadamardMultiplyBlock(input, output);

        const size_t sample_count = output.SampleCount();
        for (size_t rotation = 0; rotation < lfos_.size(); ++rotation)
        {
            auto first_channel = output.GetChannelSpan(static_cast<uint32_t>(2U * rotation));
            auto second_channel = output.GetChannelSpan(static_cast<uint32_t>((2U * rotation) + 1U));
            float phase = lfo_phases_[rotation];
            const float phase_increment = lfos_[rotation].GetFrequency();
            const float phase_offset = lfos_[rotation].GetPhaseOffset();
            const float amplitude = lfos_[rotation].GetAmplitudeNonBlocking();
            for (size_t sample = 0; sample < sample_count; ++sample)
            {
                float sine = 0.0F;
                float cosine = 0.0F;
                phase += phase_increment;
                phase -= std::floor(phase);
                const float modulation = SineTableLookup(phase + phase_offset) * amplitude;
                const float theta = base_angles_[rotation] + (std::numbers::pi_v<float> * modulation);
                SinCosUnit(theta, sine, cosine);

                const float first = first_channel[sample];
                const float second = second_channel[sample];
                first_channel[sample] = (cosine * first) - (sine * second);
                second_channel[sample] = (sine * first) + (cosine * second);
            }
            lfo_phases_[rotation] = phase;
        }

        HadamardMultiplyBlock(output, output);
        return;
    }

    const size_t sample_count = output.SampleCount();
    AudioBuffer scratch_buffer(kChunkSize, order_, scratch_);
    for (size_t block_start = 0; block_start < sample_count; block_start += kChunkSize)
    {
        const size_t block_size = std::min(kChunkSize, sample_count - block_start);
        const auto input_block = input.Offset(static_cast<uint32_t>(block_start), static_cast<uint32_t>(block_size));
        auto output_block = output.Offset(static_cast<uint32_t>(block_start), static_cast<uint32_t>(block_size));
        DenseMatVecBlock(schur_basis_, true, input_block, scratch_buffer, order_, block_size);

        for (size_t rotation = 0; rotation < lfos_.size(); ++rotation)
        {
            const uint32_t first_channel = rotation_starts_[rotation];
            auto first_channel_data = scratch_buffer.GetChannelSpan(first_channel).first(block_size);
            auto second_channel_data = scratch_buffer.GetChannelSpan(first_channel + 1U).first(block_size);
            float phase = lfo_phases_[rotation];
            const float phase_increment = lfos_[rotation].GetFrequency();
            const float phase_offset = lfos_[rotation].GetPhaseOffset();
            const float amplitude = lfos_[rotation].GetAmplitudeNonBlocking();
            for (size_t sample = 0; sample < block_size; ++sample)
            {
                float sine = 0.0F;
                float cosine = 0.0F;
                phase += phase_increment;
                phase -= std::floor(phase);
                const float modulation = SineTableLookup(phase + phase_offset) * amplitude;
                const float theta = base_angles_[rotation] + (std::numbers::pi_v<float> * modulation);
                SinCosUnit(theta, sine, cosine);
                const float first = first_channel_data[sample];
                const float second = second_channel_data[sample];
                first_channel_data[sample] = (cosine * first) - (sine * second);
                second_channel_data[sample] = (sine * first) + (cosine * second);
            }
            lfo_phases_[rotation] = phase;
        }

        for (uint32_t channel = 0; channel < order_; ++channel)
        {
            if (scalar_signs_[channel] < 0.0F)
            {
                auto channel_data = scratch_buffer.GetChannelSpan(channel).first(block_size);
                for (size_t sample = 0; sample < block_size; ++sample)
                {
                    channel_data[sample] = -channel_data[sample];
                }
            }
        }

        DenseMatVecBlock(schur_basis_, false, scratch_buffer, output_block, order_, block_size);
    }
}

uint32_t TimeVaryingFeedbackMatrix::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return order_;
}

uint32_t TimeVaryingFeedbackMatrix::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return order_;
}

uint32_t TimeVaryingFeedbackMatrix::RotationBlockCount() const noexcept SFFDN_NONBLOCKING
{
    return static_cast<uint32_t>(rotation_starts_.size());
}

bool TimeVaryingFeedbackMatrix::GetMatrix(std::span<float> matrix, uint64_t sample_index) const
{
    const size_t element_count = static_cast<size_t>(order_) * static_cast<size_t>(order_);
    if (matrix.size() != element_count)
    {
        return false;
    }

    std::vector<float> angles(base_angles_.size(), 0.0F);
    for (size_t rotation = 0; rotation < angles.size(); ++rotation)
    {
        // Process advances each LFO before it uses it, so the sample at index n sees phase (n + 1) * increment.
        // Accumulate in double before wrapping: a normalized increment near 1e-5 multiplied by a large sample index
        // loses every meaningful fractional bit in float32.
        const double advanced =
            static_cast<double>(lfos_[rotation].GetFrequency()) * (static_cast<double>(sample_index) + 1.0);
        const auto phase = static_cast<float>(advanced - std::floor(advanced));
        const float modulation =
            SineTableLookup(phase + lfos_[rotation].GetPhaseOffset()) * lfos_[rotation].GetAmplitudeNonBlocking();
        angles[rotation] = base_angles_[rotation] + (std::numbers::pi_v<float> * modulation);
    }

    // Column j of the matrix is A * e_j, so pushing an order x order identity through the same transform chain
    // Process uses, with the angles held fixed, materializes every column in a single pass.
    std::vector<float> ping(element_count, 0.0F);
    std::vector<float> pong(element_count, 0.0F);
    AudioBuffer ping_buffer(order_, order_, ping);
    AudioBuffer pong_buffer(order_, order_, pong);
    for (uint32_t channel = 0; channel < order_; ++channel)
    {
        ping_buffer.GetChannelSpan(channel)[channel] = 1.0F;
    }

    const AudioBuffer* result = nullptr;
    if (mode_ == TimeVaryingMatrixMode::Hadamard)
    {
        HadamardMultiplyBlock(ping_buffer, pong_buffer);
        ApplyFixedRotationsBlock(pong_buffer, rotation_starts_, angles, order_);
        HadamardMultiplyBlock(pong_buffer, pong_buffer);
        result = &pong_buffer;
    }
    else
    {
        // DenseMatVecBlock accumulates into its destination, so the two calls must not alias.
        DenseMatVecBlock(schur_basis_, true, ping_buffer, pong_buffer, order_, order_);
        ApplyFixedRotationsBlock(pong_buffer, rotation_starts_, angles, order_);
        for (uint32_t channel = 0; channel < order_; ++channel)
        {
            if (scalar_signs_[channel] < 0.0F)
            {
                auto channel_data = pong_buffer.GetChannelSpan(channel);
                for (uint32_t sample = 0; sample < order_; ++sample)
                {
                    channel_data[sample] = -channel_data[sample];
                }
            }
        }
        DenseMatVecBlock(schur_basis_, false, pong_buffer, ping_buffer, order_, order_);
        result = &ping_buffer;
    }

    for (uint32_t row = 0; row < order_; ++row)
    {
        const auto row_data = result->GetChannelSpan(row);
        for (uint32_t column = 0; column < order_; ++column)
        {
            matrix[(static_cast<size_t>(column) * order_) + row] = row_data[column];
        }
    }

    return true;
}

void TimeVaryingFeedbackMatrix::Clear()
{
    for (auto& lfo : lfos_)
    {
        lfo.ResetPhase();
    }
    std::ranges::fill(lfo_phases_, 0.0F);
}

std::unique_ptr<AudioProcessor> TimeVaryingFeedbackMatrix::Clone() const
{
    return std::make_unique<TimeVaryingFeedbackMatrix>(*this);
}

} // namespace sfFDN
