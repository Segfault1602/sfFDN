#include "sffdn/feedback_matrix.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/matrix_gallery.h"

#include "matrix_multiplication.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// #include <sanitizer/rtsan_interface.h>

#ifdef SFFDN_USE_VDSP
#include <Accelerate/Accelerate.h>
#endif

#include <Eigen/Core>

namespace sfFDN
{

ScalarFeedbackMatrix::ScalarFeedbackMatrix(const ScalarFeedbackMatrixOptions& config)
    : order_(config.matrix_size)
    , matrix_type_(config.custom_matrix ? ScalarMatrixType::Count : config.type)
{
    // Eigen lazily queries CPU cache sizes on the first dense product. Initialize that state during setup, not in the
    // audio callback.
    static_cast<void>(Eigen::l1CacheSize());

    if (config.custom_matrix)
    {
        const size_t expected = static_cast<size_t>(config.matrix_size) * config.matrix_size;
        if (config.custom_matrix->size() != expected)
        {
            throw std::invalid_argument("ScalarFeedbackMatrix: custom_matrix size must equal matrix_size^2 (got " +
                                        std::to_string(config.custom_matrix->size()) + ", expected " +
                                        std::to_string(expected) + ")");
        }
        matrix_data_ = *config.custom_matrix;
    }
    else
    {
        matrix_data_ = GenerateMatrix(config.matrix_size, config.type);
    }
}

bool ScalarFeedbackMatrix::SetMatrix(const std::span<const float> matrix)
{
    // Only accept exactly order_^2 elements; the channel count cannot change.
    const size_t expected = static_cast<size_t>(order_) * order_;
    if (matrix.size() != expected)
    {
        std::print(std::cerr, "ScalarFeedbackMatrix::SetMatrix: expected {} elements (order^2), got {}\n", expected,
                   matrix.size());
        return false;
    }
    // Update state atomically: assign first, then change type.
    matrix_data_.assign(matrix.begin(), matrix.end());
    matrix_type_ = ScalarMatrixType::Count;
    return true;
}

bool ScalarFeedbackMatrix::GetMatrix(std::span<float> matrix) const
{
    if (matrix.size() != order_ * order_)
    {
        return false;
    }
    std::ranges::copy(matrix_data_, matrix.begin());
    return true;
}

void ScalarFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == order_);

    const uint32_t col = order_;
    const uint32_t row = input.SampleCount();

    if (matrix_type_ == ScalarMatrixType::Hadamard && std::has_single_bit(order_))
    {
        HadamardMultiplyBlock(input, output);
        return;
    }

    if (matrix_type_ == ScalarMatrixType::Householder)
    {
        HouseholderMultiplyBlock(input, output);
        return;
    }

// Not using vDSP for now as it seems to be slower than Eigen
#if 0 // defined(SFFDN_USE_VDSP)
    const float* A = matrix_data_.data();
    const float* B = input.Data();
    float* C = output.Data();

    vDSP_mmul(A, 1, B, 1, C, 1, col, row, col);
#else

    const Eigen::Map<const Eigen::MatrixXf> matrix(matrix_data_.data(), col, col);

    const Eigen::Map<const Eigen::MatrixXf> input_map(input.Data(), row, col);
    Eigen::Map<Eigen::MatrixXf> output_map(output.Data(), row, col);

    // Intentional Eigen column-major trick:
    //   matrix_data_ stores A in row-major order: flat[r*N+c] = A[r,c].
    //   Eigen maps that buffer column-major, so the Eigen matrix object equals A^T.
    //   input_map(s, ch) = sample s of channel ch (channel-major AudioBuffer maps correctly as column-major).
    //   output_map = input_map * A^T, which for each sample-vector x_s gives y_s = A * x_s. ✓
    // noalias() avoids an alias-protection result temporary; Eigen may still use internal GEMM scratch storage.
    if (input.Data() != output.Data())
    {
        SFFDN_FEA_UNSAFE(output_map.noalias() = input_map * matrix;)
    }
    else
    {
        // TODO(Phase 2): use preallocated scratch storage for aliased matrix multiplication.
        SFFDN_FEA_UNSAFE({
            SFFDN_RTSAN_SCOPED_DISABLER(rtsan_disabler);
            output_map = input_map * matrix;
        })
    }
#endif
}

uint32_t ScalarFeedbackMatrix::GetSize() const
{
    return order_;
}

float ScalarFeedbackMatrix::GetCoefficient(uint32_t row, uint32_t col) const
{
    return matrix_data_[(row * order_) + col];
}

uint32_t ScalarFeedbackMatrix::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return order_;
}

uint32_t ScalarFeedbackMatrix::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return order_;
}

void ScalarFeedbackMatrix::Clear()
{
    // No-op for scalar feedback matrix
}

std::unique_ptr<AudioProcessor> ScalarFeedbackMatrix::Clone() const
{
    auto clone = std::make_unique<ScalarFeedbackMatrix>(*this);
    return clone;
}

} // namespace sfFDN