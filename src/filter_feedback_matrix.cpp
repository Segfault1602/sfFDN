#include "sffdn/filter_feedback_matrix.h"

#include <cassert>
#include <iostream>
#include <print>

namespace sfFDN
{
FilterFeedbackMatrix::FilterFeedbackMatrix(uint32_t N)
    : N_(N)
{
    delays_.clear();
    matrix_.clear();
}

void FilterFeedbackMatrix::Clear()
{
    for (auto& delay : delays_)
    {
        delay.Clear();
    }
}

void FilterFeedbackMatrix::ConstructMatrix(std::span<const uint32_t> delays,
                                           std::span<const ScalarFeedbackMatrix> mixing_matrices)
{
    const uint32_t num_stages = (delays.size() / N_) - 1;
    assert(mixing_matrices.size() == num_stages);

    delays_.reserve(num_stages + 1);
    matrix_.reserve(mixing_matrices.size());
    for (uint32_t i = 0; i < num_stages + 1; ++i)
    {
        auto stage_delays = delays.subspan(i * N_, N_);
        delays_.emplace_back(stage_delays, 128); // TODO: Adjust block size as needed
    }

    for (size_t i = 0; i < mixing_matrices.size(); ++i)
    {
        matrix_.emplace_back(mixing_matrices[i]);
    }
}

void FilterFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == N_);
    assert(output.ChannelCount() == N_);

    if (!delays_.empty())
    {
        // Apply first stage
        delays_[0].Process(input, output);
        matrix_[0].Process(output, output);

        for (uint32_t i = 1; i < matrix_.size(); ++i)
        {
            delays_[i].Process(output, output);
            matrix_[i].Process(output, output);
        }

        // Apply last delays
        // delays_.back().Process(output, output);
    }
}

void FilterFeedbackMatrix::PrintInfo() const
{
    std::println("FilterFeedbackMatrix Info:");
    std::println("Number of stages: {}", delays_.size());
    for (const auto& delay : delays_)
    {
        auto delays = delay.GetDelays();
        std::println("Delays: [");
        for (size_t i = 0; i < delays.size(); ++i)
        {
            std::print("{}", delays[i]);
            if (i < delays.size() - 1)
            {
                std::print(", ");
            }
            std::println("]");
        }
    }
}

std::unique_ptr<FilterFeedbackMatrix> MakeFilterFeedbackMatrix(const CascadedFeedbackMatrixInfo& info)
{
    if (info.delays.size() % info.N != 0)
    {
        std::println(std::cerr, "Delays size must be a multiple of N.");
        return nullptr;
    }

    if (info.delays.size() / info.N != info.K + 1)
    {
        std::println(std::cerr, "Delays size does not match the expected number of stages.");
        return nullptr;
    }

    if (info.matrices.size() != info.K * info.N * info.N)
    {
        std::println(std::cerr, "Matrices size does not match the expected size for K stages.");
        return nullptr;
    }

    std::vector<sfFDN::ScalarFeedbackMatrix> feedback_matrices;
    for (size_t i = 0; i < info.K; i++)
    {
        std::span<const float> matrix_span(info.matrices.data() + i * info.N * info.N, info.N * info.N);
        sfFDN::ScalarFeedbackMatrix feedback_matrix(info.N);
        feedback_matrix.SetMatrix(matrix_span);
        feedback_matrices.push_back(feedback_matrix);
    }
    auto ffm = std::make_unique<sfFDN::FilterFeedbackMatrix>(info.N);
    ffm->ConstructMatrix(info.delays, feedback_matrices);
    return ffm;
}

} // namespace sfFDN