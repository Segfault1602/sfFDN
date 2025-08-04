#include "sffdn/delay_matrix.h"

#include <cassert>
#include <iostream>
#include <mdspan>
#include <print>

namespace sfFDN
{

DelayMatrix::DelayMatrix(uint32_t N, std::span<const uint32_t> delays, ScalarFeedbackMatrix mixing_matrix)
    : FeedbackMatrix(N)
    , delays_(N)
    , N_(N)
    , mixing_matrix_(mixing_matrix)
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

    signal_matrix = Eigen::MatrixXf::Zero(N_, N_);
}

void DelayMatrix::Clear()
{
    delays_.Clear();
}

void DelayMatrix::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == delays_.InputChannelCount());

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
                signal_matrix(j, k) = delay_lines_[j].TapOut(delay_mdspan[j, k]);
            }
        }

        auto result = signal_matrix.cwiseProduct(matrix_).colwise().sum();

        for (size_t j = 0; j < output.ChannelCount(); ++j)
        {
            output.GetChannelSpan(j)[i] = result[j];
        }
    }
}

void DelayMatrix::PrintInfo() const
{
    std::println("DelayMatrix Info:");
    std::println("Delays: [");
    auto delays = delays_.GetDelays();
    for (size_t i = 0; i < delays.size(); ++i)
    {
        std::print("{}", delays[i]);
        if (i < delays.size() - 1)
        {
            std::print(", ");
        }
    }
    std::println("]");
    std::print("Mixing Matrix: ");
    mixing_matrix_.Print();
}

} // namespace sfFDN