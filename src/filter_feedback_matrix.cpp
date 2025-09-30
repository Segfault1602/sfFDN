#include "sffdn/filter_feedback_matrix.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/matrix_gallery.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <print>
#include <span>
#include <vector>

namespace
{
constexpr uint32_t kDefaultBlockSize = 1024; // Default block size for delay banks
}

namespace sfFDN
{
FilterFeedbackMatrix::FilterFeedbackMatrix(uint32_t channel_count)
    : channel_count_(channel_count)
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
    const uint32_t num_stages = (delays.size() / channel_count_) - 1;
    assert(mixing_matrices.size() == num_stages);

    delays_.reserve(num_stages + 1);
    matrix_.reserve(mixing_matrices.size());
    for (uint32_t i = 0; i < num_stages + 1; ++i)
    {
        auto stage_delays = delays.subspan(i * channel_count_, channel_count_);
        delays_.emplace_back(stage_delays, kDefaultBlockSize);
    }

    for (const auto& mixing_matrix : mixing_matrices)
    {
        matrix_.emplace_back(mixing_matrix);
    }
}

void FilterFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == channel_count_);
    assert(output.ChannelCount() == channel_count_);

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
        for (auto i = 0u; i < delays.size(); ++i)
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

bool FilterFeedbackMatrix::GetFirstMatrix(std::span<float> matrix) const
{
    if (matrix.size() != channel_count_ * channel_count_)
    {
        return false;
    }

    if (matrix_.empty())
    {
        return false;
    }

    return matrix_[0].GetMatrix(matrix);
}

std::unique_ptr<FilterFeedbackMatrix> MakeFilterFeedbackMatrix(const CascadedFeedbackMatrixInfo& info)
{
    if (info.delays.size() % info.channel_count != 0)
    {
        std::println(std::cerr, "Delays size must be a multiple of channel_count.");
        return nullptr;
    }

    if (info.delays.size() / info.channel_count != info.stage_count + 1)
    {
        std::println(std::cerr, "Delays size does not match the expected number of stages.");
        return nullptr;
    }

    if (info.matrices.size() != info.stage_count * info.channel_count * info.channel_count)
    {
        std::println(std::cerr, "Matrices size does not match the expected size for stage_count stages.");
        return nullptr;
    }

    std::vector<sfFDN::ScalarFeedbackMatrix> feedback_matrices;
    auto all_matrices_span = std::span(info.matrices);
    for (auto i = 0u; i < info.stage_count; i++)
    {
        std::span<const float> matrix_span = all_matrices_span.subspan(i * info.channel_count * info.channel_count,
                                                                       info.channel_count * info.channel_count);
        sfFDN::ScalarFeedbackMatrix feedback_matrix(info.channel_count);
        feedback_matrix.SetMatrix(matrix_span);
        feedback_matrices.push_back(feedback_matrix);
    }
    auto ffm = std::make_unique<sfFDN::FilterFeedbackMatrix>(info.channel_count);
    ffm->ConstructMatrix(info.delays, feedback_matrices);
    return ffm;
}

std::unique_ptr<AudioProcessor> FilterFeedbackMatrix::Clone() const
{
    auto clone = std::make_unique<FilterFeedbackMatrix>(channel_count_);

    for (const auto& delay : delays_)
    {
        clone->delays_.emplace_back(delay.GetDelays(), kDefaultBlockSize);
    }

    clone->matrix_ = matrix_;
    return clone;
}

} // namespace sfFDN