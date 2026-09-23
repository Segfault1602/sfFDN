#include "sffdn/feedback_matrix.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/matrix_gallery.h"

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

namespace sfFDN
{

ScalarFeedbackMatrix::ScalarFeedbackMatrix(const ScalarFeedbackMatrixOptions& config)
    : order_(detail::RequireValidOptions(config).MatrixSize())
    , matrix_type_(std::visit(overloaded{
                                  [](const GeneratedMatrixOptions& source) { return GetMatrixType(source.generator); },
                                  [](const MatrixData&) { return ScalarMatrixType::Count; },
                              },
                              config.source))
    , scratch_data_(static_cast<size_t>(order_) * kDenseMatrixScratchFrames)
{
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

    if (input.SampleCount() == 0)
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

    MultiplyDenseMatrix(input, output, matrix_data_, scratch_data_);
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