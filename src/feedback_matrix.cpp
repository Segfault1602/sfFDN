#include "sffdn/feedback_matrix.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/matrix_gallery.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <print>
#include <span>
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
{
    if (config.custom_matrix)
    {
        matrix_data_ = *config.custom_matrix;
    }
    else
    {
        matrix_data_ = GenerateMatrix(config.matrix_size, config.type);
    }
}

ScalarFeedbackMatrix::~ScalarFeedbackMatrix() = default;

bool ScalarFeedbackMatrix::SetMatrix(const std::span<const float> matrix)
{
    auto order = static_cast<uint32_t>(std::sqrt(matrix.size()));
    if (order * order != matrix.size() || order == 0)
    {
        std::print(std::cerr, "Only square matrices are supported!\n");
        return false;
    }
    matrix_data_ = std::vector<float>(matrix.begin(), matrix.end());
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

void ScalarFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == order_);

    const uint32_t col = order_;
    const uint32_t row = input.SampleCount();

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

    // The input and output buffers must not overlap
    // This is a requirement to avoid memory allocation in Eigen by using noalias()
    if (input.Data() != output.Data())
    {
        output_map.noalias() = input_map * matrix;
    }
    else
    {
        // __rtsan::ScopedDisabler d;
        // I think this path is only used for the FilterFeedbackMatrix, but could be fixed by using a temporary
        // buffer
        output_map = input_map * matrix;
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

uint32_t ScalarFeedbackMatrix::InputChannelCount() const
{
    return order_;
}

uint32_t ScalarFeedbackMatrix::OutputChannelCount() const
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