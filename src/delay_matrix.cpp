#include "sffdn/delay_matrix.h"

#include <cassert>
#include <iostream>
#include <mdspan>
#include <print>

#include <Eigen/Core>

#include "sffdn/delay.h"

namespace sfFDN
{

class DelayMatrix::DelayMatrixImpl
{
  public:
    DelayMatrixImpl(size_t N, std::span<const uint32_t> delays, ScalarFeedbackMatrix mixing_matrix)
        : N_(N)
    {
        assert(delays.size() == N * N);

        delay_values_.assign(delays.begin(), delays.end());
        delay_lines_.reserve(N);

        std::vector<uint32_t> max_delays(N, 0);
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                max_delays[i] = std::max(max_delays[i], delays[i * N + j]);
            }

            delay_lines_.emplace_back(max_delays[i], max_delays[i]);
        }

        matrix_ = Eigen::MatrixXf::Zero(N, N);
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                matrix_(i, j) = mixing_matrix.GetCoefficient(i, j);
            }
        }

        signal_matrix_ = Eigen::MatrixXf::Zero(N_, N_);
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

        auto delay_mdspan = std::mdspan(delay_values_.data(), N_, N_);

        for (size_t i = 0; i < input.SampleCount(); ++i)
        {
            // Add input samples to the delay lines
            for (size_t j = 0; j < input.ChannelCount(); ++j)
            {
                delay_lines_[j].Tick(input.GetChannelSpan(j)[i]);
            }

            // Fill the signal matrix with the current outputs from the delay lines
            for (size_t j = 0; j < N_; ++j)
            {
                for (size_t k = 0; k < N_; ++k)
                {
                    signal_matrix_(j, k) = delay_lines_[j].TapOut(delay_mdspan[j, k]);
                }
            }

            auto result = signal_matrix_.cwiseProduct(matrix_).colwise().sum();

            for (size_t j = 0; j < output.ChannelCount(); ++j)
            {
                output.GetChannelSpan(j)[i] = result[j];
            }
        }
    }

    uint32_t InputChannelCount() const
    {
        return N_;
    }

    uint32_t OutputChannelCount() const
    {
        return N_;
    }

    void PrintInfo() const
    {
        std::println("DelayMatrix Info:");
        std::println("Delays:");
        Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>> delay_matrix(delay_values_.data(), N_,
                                                                                               N_);
        std::cout << delay_matrix << std::endl;

        std::println("Mixing Matrix:");
        std::cout << signal_matrix_ << std::endl;
    }

  private:
    size_t N_;
    std::vector<Delay> delay_lines_;
    std::vector<uint32_t> delay_values_;
    Eigen::MatrixXf matrix_;
    Eigen::MatrixXf signal_matrix_;
};

DelayMatrix::DelayMatrix(uint32_t N, std::span<const uint32_t> delays, ScalarFeedbackMatrix mixing_matrix)
{
    impl_ = std::make_unique<DelayMatrixImpl>(N, delays, mixing_matrix);
}

DelayMatrix::~DelayMatrix() = default;

void DelayMatrix::Clear()
{
    impl_->Clear();
}

void DelayMatrix::Process(const AudioBuffer& input, AudioBuffer& output)
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

} // namespace sfFDN