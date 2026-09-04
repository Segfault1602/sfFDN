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
        assert(delays.size() == static_cast<size_t>(order) * order);

        delay_values_.assign(delays.begin(), delays.end());
        delay_lines_.reserve(order);

        // matrix_(destination, source): gain from source channel to destination channel.
        matrix_ = Eigen::MatrixXf::Zero(order, order);
        for (auto dest = 0u; dest < order; ++dest)
        {
            for (auto src = 0u; src < order; ++src)
            {
                matrix_(dest, src) = mixing_matrix.GetCoefficient(dest, src);
            }
        }

        // delay_values_[dest * order + src] = tap depth for path src→dest.
        // Size each source delay line to the deepest tap used across all destinations.
        std::vector<uint32_t> max_delays(order, 0);
        for (auto src = 0u; src < order; ++src)
        {
            for (auto dest = 0u; dest < order; ++dest)
            {
                max_delays[src] = std::max(max_delays[src], delay_values_[(dest * order) + src]);
            }
            delay_lines_.emplace_back(max_delays[src], max_delays[src]);
        }
    }

    void Clear()
    {
        for (auto& delay : delay_lines_)
        {
            delay.Clear();
        }
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
    {
        assert(input.SampleCount() == output.SampleCount());
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == delay_lines_.size());

        for (auto s = 0u; s < input.SampleCount(); ++s)
        {
            // Push current input samples into each source delay line.
            for (auto src = 0u; src < order_; ++src)
            {
                delay_lines_[src].Tick(input.GetChannelSpan(src)[s]);
            }

            // Accumulate: output[dest] = sum_src A[dest,src] * delayed_input[src, delays[dest*N+src]]
            for (auto dest = 0u; dest < order_; ++dest)
            {
                float acc = 0.0f;
                for (auto src = 0u; src < order_; ++src)
                {
                    acc += matrix_(dest, src) * delay_lines_[src].TapOut(delay_values_[(dest * order_) + src]);
                }
                output.GetChannelSpan(dest)[s] = acc;
            }
        }
    }

    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING
    {
        return order_;
    }

    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING
    {
        return order_;
    }

    void PrintInfo() const
    {
        std::println("DelayMatrix Info:");
        std::println("Order: {}", order_);
        std::println("Delays [row=dest, col=src, delays[dest*N+src]]:");
        for (auto dest = 0u; dest < order_; ++dest)
        {
            for (auto src = 0u; src < order_; ++src)
            {
                std::print("{:6}", delay_values_[(dest * order_) + src]);
            }
            std::println("");
        }
        std::println("Mixing matrix [row=dest, col=src, matrix_(dest,src)]:");
        std::cout << matrix_ << '\n';
    }

    std::unique_ptr<DelayMatrixImpl> Clone() const
    {
        return std::make_unique<DelayMatrixImpl>(*this);
    }

    nlohmann::json ToJson() const
    {
        nlohmann::json j;
        j["type"] = "DelayMatrix";
        j["order"] = order_;
        j["delays"] = delay_values_;

        std::vector<float> matrix_data;
        matrix_data.reserve(order_ * order_);
        for (auto dest = 0u; dest < order_; ++dest)
        {
            for (auto src = 0u; src < order_; ++src)
            {
                matrix_data.push_back(matrix_(dest, src));
            }
        }

        j["matrix"] = matrix_data;
        return j;
    }

  private:
    uint32_t order_;
    std::vector<Delay> delay_lines_;
    std::vector<uint32_t> delay_values_;
    Eigen::MatrixXf matrix_;
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

void DelayMatrix::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    impl_->Process(input, output);
}

uint32_t DelayMatrix::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return impl_->InputChannelCount();
}

uint32_t DelayMatrix::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
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