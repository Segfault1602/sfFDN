#include "sffdn/delay_matrix.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay.h"
#include "sffdn/feedback_matrix.h"

#include <Eigen/Core>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <mdspan>
#include <memory>
#include <print>
#include <span>
#include <utility>
#include <vector>

namespace sfFDN
{

class DelayMatrix::DelayMatrixImpl
{
  public:
    DelayMatrixImpl(uint32_t order, std::span<const uint32_t> delays, const ScalarFeedbackMatrix& mixing_matrix)
        : order_(order)
    {
        assert(delays.size() == order * order);

        delay_values_.assign(delays.begin(), delays.end());
        delay_lines_.reserve(order);

        matrix_ = Eigen::MatrixXf::Zero(order, order);
        for (auto i = 0u; i < order; ++i)
        {
            for (auto j = 0u; j < order; ++j)
            {
                matrix_(i, j) = mixing_matrix.GetCoefficient(i, j);
            }
        }

        std::vector<uint32_t> max_delays(order, 0);
        for (auto i = 0u; i < order; ++i)
        {
            for (auto j = 0u; j < order; ++j)
            {
                max_delays[i] = std::max(max_delays[i], delays[(i * order) + j]);
            }

            delay_lines_.emplace_back(max_delays[i], max_delays[i]);
        }

        signal_matrix_ = Eigen::MatrixXf::Zero(order_, order_);
    }

    void Clear()
    {
        for (auto& delay : delay_lines_)
        {
            delay.Clear();
        }
    }

    void Process(const AudioBuffer& input, AudioBuffer& output)
    {
        assert(input.SampleCount() == output.SampleCount());
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == delay_lines_.size());

        auto delay_mdspan = std::mdspan(delay_values_.data(), order_, order_);

        for (auto i = 0u; i < input.SampleCount(); ++i)
        {
            // Add input samples to the delay lines
            for (auto j = 0u; j < input.ChannelCount(); ++j)
            {
                delay_lines_[j].Tick(input.GetChannelSpan(j)[i]);
            }

            // Fill the signal matrix with the current outputs from the delay lines
            for (auto j = 0u; j < order_; ++j)
            {
                for (auto k = 0u; k < order_; ++k)
                {
                    signal_matrix_(j, k) = delay_lines_[j].TapOut(delay_mdspan[j, k]);
                }
            }

            auto result = signal_matrix_.cwiseProduct(matrix_).colwise().sum();

            for (auto j = 0u; j < output.ChannelCount(); ++j)
            {
                output.GetChannelSpan(j)[i] = result[j];
            }
        }
    }

    uint32_t InputChannelCount() const
    {
        return order_;
    }

    uint32_t OutputChannelCount() const
    {
        return order_;
    }

    void PrintInfo() const
    {
        std::println("DelayMatrix Info:");
        std::println("Delays:");
        Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>> delay_matrix(delay_values_.data(),
                                                                                               order_, order_);
        std::cout << delay_matrix << '\n';

        std::println("Mixing Matrix:");
        std::cout << signal_matrix_ << "\n";
    }

    std::unique_ptr<DelayMatrixImpl> Clone() const
    {
        return std::make_unique<DelayMatrixImpl>(*this);
    }

  private:
    uint32_t order_;
    std::vector<Delay> delay_lines_;
    std::vector<uint32_t> delay_values_;
    Eigen::MatrixXf matrix_;
    Eigen::MatrixXf signal_matrix_;
};

DelayMatrix::DelayMatrix(uint32_t order, std::span<const uint32_t> delays, const ScalarFeedbackMatrix& mixing_matrix)
{
    impl_ = std::make_unique<DelayMatrixImpl>(order, delays, mixing_matrix);
}

DelayMatrix::~DelayMatrix() = default;

DelayMatrix::DelayMatrix(const DelayMatrix& other)
    : impl_(other.impl_->Clone())
{
}

DelayMatrix& DelayMatrix::operator=(const DelayMatrix& other)
{
    if (this != &other)
    {
        impl_ = other.impl_->Clone();
    }
    return *this;
}

DelayMatrix::DelayMatrix(DelayMatrix&& other) noexcept
    : impl_(std::move(other.impl_))
{
}

DelayMatrix& DelayMatrix::operator=(DelayMatrix&& other) noexcept
{
    impl_ = std::move(other.impl_);
    return *this;
}

void DelayMatrix::Clear()
{
    impl_->Clear();
}

void DelayMatrix::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    impl_->Process(input, output);
}

uint32_t DelayMatrix::InputChannelCount() const
{
    return impl_->InputChannelCount();
}

uint32_t DelayMatrix::OutputChannelCount() const
{
    return impl_->OutputChannelCount();
}

void DelayMatrix::PrintInfo() const
{
    impl_->PrintInfo();
}

std::unique_ptr<AudioProcessor> DelayMatrix::Clone() const
{
    auto clone = std::make_unique<DelayMatrix>(*this);
    return clone;
}

} // namespace sfFDN