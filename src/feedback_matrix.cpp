#include "sffdn/feedback_matrix.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/matrix_gallery.h"

#include "audio_buffer_alias.h"
#include "matrix_multiplication.h"
#include "processor_option_validation.h"

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

namespace
{

void MultiplyDenseMatrix(const float* input, float* output, const float* matrix_data, uint32_t row, uint32_t col,
                         const Eigen::OuterStride<>& input_stride,
                         const Eigen::OuterStride<>& output_stride) noexcept SFFDN_NONBLOCKING
{
    const Eigen::Map<const Eigen::MatrixXf> matrix(matrix_data, col, col);
    const Eigen::Map<const Eigen::MatrixXf, Eigen::Unaligned, Eigen::OuterStride<>> input_map(input, row, col,
                                                                                              input_stride);
    Eigen::Map<Eigen::MatrixXf, Eigen::Unaligned, Eigen::OuterStride<>> output_map(output, row, col, output_stride);
    SFFDN_FEA_UNSAFE(output_map.noalias() = input_map * matrix;)
}

} // namespace

namespace sfFDN
{

ScalarFeedbackMatrix::ScalarFeedbackMatrix(const ScalarFeedbackMatrixOptions& config)
    : order_(detail::RequireValidOptions(config).MatrixSize())
    , matrix_type_(std::visit(overloaded{
                                  [](const GeneratedMatrixOptions& source) { return GetMatrixType(source.generator); },
                                  [](const MatrixData&) { return ScalarMatrixType::Count; },
                              },
                              config.source))
    , scratch_data_(static_cast<size_t>(order_) * kScratchTileFrames)
{
    // Eigen lazily queries CPU cache sizes on the first dense product. Initialize that state during setup, not in the
    // audio callback.
    static_cast<void>(Eigen::l1CacheSize());

    std::visit(
        overloaded{
            [this](const GeneratedMatrixOptions& source) {
                matrix_data_ = GenerateMatrix(source.matrix_size, source.generator, source.rng_seed);
            },
            [this](const MatrixData& source) { matrix_data_.assign(source.Values().begin(), source.Values().end()); },
        },
        config.source);
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

    if (row == 0)
    {
        return;
    }

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

    // Intentional Eigen column-major trick:
    //   matrix_data_ stores A in row-major order: flat[r*N+c] = A[r,c].
    //   Eigen maps that buffer column-major, so the Eigen matrix object equals A^T.
    //   Logical input maps use samples as rows and channels as columns.
    //   Multiplication by A^T therefore gives y_s = A * x_s for each sample-vector x_s. ✓
    // noalias() avoids an alias-protection result temporary; Eigen may still use internal GEMM scratch storage.
    // Partial overlap is outside the supported contract: Debug asserts, and Release falls through to the tiled path,
    // which is defined but numerically unspecified for that case.
    const AudioBufferAlias alias = ClassifyAudioBufferAlias(input, output);
    assert(alias != AudioBufferAlias::Partial);

    const float* input_base = input.GetChannelSpan(0).data();
    float* output_base = output.GetChannelSpan(0).data();
    const Eigen::OuterStride<> input_stride(input.ChannelStride());
    const Eigen::OuterStride<> output_stride(output.ChannelStride());

    if (alias == AudioBufferAlias::Disjoint)
    {
        MultiplyDenseMatrix(input_base, output_base, matrix_data_.data(), row, col, input_stride, output_stride);
        return;
    }

    for (uint32_t tile_begin = 0; tile_begin < row; tile_begin += kScratchTileFrames)
    {
        const uint32_t tile_size = std::min(kScratchTileFrames, row - tile_begin);
        for (uint32_t channel = 0; channel < col; ++channel)
        {
            const auto channel_input = input.GetChannelSpan(channel);
            const auto tile_input = channel_input.subspan(tile_begin, tile_size);
            const auto scratch_output =
                std::span(scratch_data_).subspan(static_cast<size_t>(channel) * kScratchTileFrames, tile_size);
            std::ranges::copy(tile_input, scratch_output.begin());
        }

        MultiplyDenseMatrix(scratch_data_.data(), output.GetChannelSpan(0).subspan(tile_begin).data(),
                            matrix_data_.data(), tile_size, col, Eigen::OuterStride<>(kScratchTileFrames),
                            output_stride);
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