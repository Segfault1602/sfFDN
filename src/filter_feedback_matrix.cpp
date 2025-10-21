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
FilterFeedbackMatrix::FilterFeedbackMatrix(const CascadedFeedbackMatrixInfo& info)
    : channel_count_(info.channel_count)
{
    delaybanks_.reserve(info.stage_count);
    matrix_.reserve(info.stage_count + 1);

    assert(info.delays.size() == info.stage_count);
    assert(info.matrices.size() == info.stage_count + 1);

    for (uint32_t i = 0; i < info.delays.size(); ++i)
    {
        auto stage_delays = info.delays[i];
        delaybanks_.emplace_back(stage_delays, kDefaultBlockSize);
    }

    for (uint32_t i = 0; i < info.matrices.size(); ++i)
    {
        const std::vector<float>& matrix = info.matrices[i];
        matrix_.emplace_back(channel_count_, matrix);
    }
}

FilterFeedbackMatrix::FilterFeedbackMatrix(const FilterFeedbackMatrix& other)
    : channel_count_(other.channel_count_)
    , delaybanks_(other.delaybanks_)
    , matrix_(other.matrix_)
{
}

FilterFeedbackMatrix& FilterFeedbackMatrix::operator=(const FilterFeedbackMatrix& other)
{
    if (this != &other)
    {
        channel_count_ = other.channel_count_;
        delaybanks_ = other.delaybanks_;
        matrix_ = other.matrix_;
    }
    return *this;
}

FilterFeedbackMatrix::FilterFeedbackMatrix(FilterFeedbackMatrix&& other) noexcept
    : channel_count_(other.channel_count_)
    , delaybanks_(std::move(other.delaybanks_))
    , matrix_(std::move(other.matrix_))
{
}

FilterFeedbackMatrix& FilterFeedbackMatrix::operator=(FilterFeedbackMatrix&& other) noexcept
{
    if (this != &other)
    {
        channel_count_ = other.channel_count_;
        delaybanks_ = std::move(other.delaybanks_);
        matrix_ = std::move(other.matrix_);
    }
    return *this;
}

void FilterFeedbackMatrix::Clear()
{
    for (auto& delay : delaybanks_)
    {
        delay.Clear();
    }
}

void FilterFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == channel_count_);
    assert(output.ChannelCount() == channel_count_);

    matrix_[0].Process(input, output);

    assert(delaybanks_.size() + 1 == matrix_.size());
    for (auto i = 0u; i < delaybanks_.size(); ++i)
    {
        delaybanks_[i].Process(output, output);
        matrix_[i + 1].Process(output, output);
    }
}

void FilterFeedbackMatrix::PrintInfo() const
{
    std::println("FilterFeedbackMatrix Info:");
    std::println("Number of stages: {}", delaybanks_.size());
    for (const auto& delay : delaybanks_)
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

std::unique_ptr<AudioProcessor> FilterFeedbackMatrix::Clone() const
{
    auto clone = std::make_unique<FilterFeedbackMatrix>(*this);
    return clone;
}

} // namespace sfFDN