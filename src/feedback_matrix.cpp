#include "sffdn/feedback_matrix.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <print>
#include <span>
#include <utility>
#include <vector>

#include "sffdn/matrix_gallery.h"

// #include <sanitizer/rtsan_interface.h>

#ifdef SFFDN_USE_VDSP
#include <Accelerate/Accelerate.h>
#endif

#include <Eigen/Core>

namespace sfFDN
{

class ScalarFeedbackMatrix::ScalarFeedbackMatrixImpl
{
  public:
    explicit ScalarFeedbackMatrixImpl(uint32_t order, ScalarMatrixType type)
        : order_(order)
    {
        matrix_data_ = GenerateMatrix(order, type);
    }

    void SetMatrix(std::span<const float> matrix)
    {
        assert(matrix.size() == order_ * order_);
        matrix_data_ = std::vector<float>(matrix.begin(), matrix.end());
    }

    bool GetMatrix(std::span<float> matrix) const
    {
        if (matrix.size() != order_ * order_)
        {
            return false;
        }

        std::ranges::copy(matrix_data_, matrix.begin());
        return true;
    }

    void Process(const AudioBuffer& input, AudioBuffer& output)
    {
        assert(input.SampleCount() == output.SampleCount());
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == order_);

        const uint32_t col = order_;
        const uint32_t row = input.SampleCount();

#ifdef SFFDN_USE_VDSP
        // #if 0
        const float* A = matrix_data_.data();
        const float* B = input.Data();
        float* C = output.Data();

        vDSP_mmul(A, 1, B, 1, C, 1, col, row, col);
#else

        Eigen::Map<const Eigen::MatrixXf> matrix(matrix_data_.data(), col, col);

        Eigen::Map<const Eigen::MatrixXf> input_map(input.Data(), row, col);
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

    void Print() const
    {
        // std::cout << matrix_ << '\n';
    }

    uint32_t GetSize() const
    {
        return order_;
    }

    float GetCoefficient(uint32_t row, uint32_t col) const
    {
        return matrix_data_[(row * order_) + col];
    }

    std::unique_ptr<ScalarFeedbackMatrixImpl> Clone() const
    {
        return std::make_unique<ScalarFeedbackMatrixImpl>(*this);
    }

    uint32_t InputChannelCount() const
    {
        return order_;
    }

    uint32_t OutputChannelCount() const
    {
        return order_;
    }

  private:
    uint32_t order_;
    // Eigen::MatrixXf matrix_;
    std::vector<float> matrix_data_;
};

ScalarFeedbackMatrix::ScalarFeedbackMatrix(uint32_t order, ScalarMatrixType type)
{
    impl_ = std::make_unique<ScalarFeedbackMatrixImpl>(order, type);
}

ScalarFeedbackMatrix::ScalarFeedbackMatrix(uint32_t order, std::span<const float> matrix)
{
    impl_ = std::make_unique<ScalarFeedbackMatrixImpl>(order, ScalarMatrixType::Identity);
    impl_->SetMatrix(matrix);
}

ScalarFeedbackMatrix::ScalarFeedbackMatrix(const ScalarFeedbackMatrix& other)
{
    impl_ = other.impl_->Clone();
}

ScalarFeedbackMatrix& ScalarFeedbackMatrix::operator=(const ScalarFeedbackMatrix& other)
{
    if (this != &other)
    {
        impl_ = other.impl_->Clone();
    }
    return *this;
}

ScalarFeedbackMatrix::ScalarFeedbackMatrix(ScalarFeedbackMatrix&& other) noexcept
    : impl_(std::move(other.impl_))
{
}

ScalarFeedbackMatrix& ScalarFeedbackMatrix::operator=(ScalarFeedbackMatrix&& other) noexcept
{
    if (this != &other)
    {
        impl_ = std::move(other.impl_);
    }
    return *this;
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
    impl_->SetMatrix(matrix);
    return true;
}

bool ScalarFeedbackMatrix::GetMatrix(std::span<float> matrix) const
{
    return impl_->GetMatrix(matrix);
}

void ScalarFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    impl_->Process(input, output);
}

void ScalarFeedbackMatrix::Print() const
{
    impl_->Print();
}

uint32_t ScalarFeedbackMatrix::GetSize() const
{
    return impl_->GetSize();
}

float ScalarFeedbackMatrix::GetCoefficient(uint32_t row, uint32_t col) const
{
    return impl_->GetCoefficient(row, col);
}

uint32_t ScalarFeedbackMatrix::InputChannelCount() const
{
    return impl_->InputChannelCount();
}

uint32_t ScalarFeedbackMatrix::OutputChannelCount() const
{
    return impl_->OutputChannelCount();
}

void ScalarFeedbackMatrix::Clear()
{
    // No-op for scalar feedback matrix
}

std::unique_ptr<AudioProcessor> ScalarFeedbackMatrix::Clone() const
{
    auto clone = std::make_unique<ScalarFeedbackMatrix>();
    clone->impl_ = impl_->Clone();
    return clone;
}

} // namespace sfFDN