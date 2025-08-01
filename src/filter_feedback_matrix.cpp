#include "filter_feedback_matrix.h"

#include <cassert>
#include <iostream>

namespace sfFDN
{
FilterFeedbackMatrix::FilterFeedbackMatrix(uint32_t N)
    : FeedbackMatrix(N)
{
    stages_.clear();
    last_mat_ = ScalarFeedbackMatrix::Eye(N);
}

void FilterFeedbackMatrix::Clear()
{
    for (auto& stage : stages_)
    {
        stage.Clear();
    }
}

void FilterFeedbackMatrix::ConstructMatrix(std::span<uint32_t> delays, std::span<ScalarFeedbackMatrix> mixing_matrices)
{
    const uint32_t num_stages = mixing_matrices.size();
    assert(delays.size() == N_ * (num_stages - 1));

    stages_.reserve(num_stages - 1);
    for (uint32_t i = 0; i < num_stages - 1; ++i)
    {
        auto stage_delays = delays.subspan(i * N_, N_);
        stages_.emplace_back(N_, stage_delays);
        stages_[i].SetMatrix(mixing_matrices[i]);
    }

    last_mat_ = mixing_matrices[num_stages - 1];
}

void FilterFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == N_);
    assert(output.ChannelCount() == N_);

    if (!stages_.empty())
    {
        // Apply first stage
        stages_[0].Process(input, output);

        for (uint32_t i = 1; i < stages_.size(); ++i)
        {
            stages_[i].Process(output, output);
        }
    }

    // Apply last delay stage
    last_mat_.Process(output, output);
}

void FilterFeedbackMatrix::PrintInfo() const
{
    std::cout << "FilterFeedbackMatrix Info:" << std::endl;
    std::cout << "Number of stages: " << stages_.size() << std::endl;
    std::cout << "Last mixing matrix size: " << last_mat_.GetSize() << std::endl;
    for (const auto& stage : stages_)
    {
        stage.PrintInfo();
    }
    last_mat_.Print();
}
} // namespace sfFDN