#include "filter_feedback_matrix.h"

#include <iostream>
#include <mdspan>

namespace fdn
{
FilterFeedbackMatrix::FilterFeedbackMatrix(size_t N, size_t K)
    : N_(N)
    , K_(K)
{
    stages_.reserve(K - 1);

    std::vector<size_t> delays(N, 0);

    for (size_t i = 0; i < K - 1; ++i)
    {
        stages_.emplace_back(N, delays);
    }
}

void FilterFeedbackMatrix::Clear()
{
    for (auto& stage : stages_)
    {
        stage.Clear();
    }
}

void FilterFeedbackMatrix::SetDelays(std::span<size_t> delays)
{
    assert(delays.size() == N_ * (K_ - 1));
    assert(stages_.size() == K_ - 1);

    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto stage_delays = delays.subspan(i * N_, N_);
        stages_[i].SetDelays(stage_delays);
    }
}

void FilterFeedbackMatrix::SetMatrices(std::span<MixMat> mixing_matrices)
{
    assert(mixing_matrices.size() == K_);
    assert(stages_.size() == K_ - 1);

    for (size_t i = 0; i < stages_.size(); ++i)
    {
        stages_[i].SetMatrix(mixing_matrices[i]);
    }

    last_mat_ = mixing_matrices[K_ - 1];
}

void FilterFeedbackMatrix::Tick(std::span<const float> input, std::span<float> output)
{
    assert(input.size() == output.size());
    assert(input.size() % N_ == 0);

    if (!stages_.empty())
    {
        // Apply first stage
        stages_[0].Tick(input, output);

        for (size_t i = 1; i < stages_.size(); ++i)
        {
            stages_[i].Tick(output, output);
        }
    }

    // Apply last delay stage
    last_mat_.Tick(output, output);
}

void FilterFeedbackMatrix::DumpDelays() const
{
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        std::cout << "Stage " << i << ": ";

        stages_[i].DumpDelays();
    }
}

} // namespace fdn