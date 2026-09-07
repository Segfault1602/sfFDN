#include "sffdn/channel_matrix.h"

#include "array_math.h"
#include "processor_option_validation.h"
#include "sffdn/audio_buffer.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#ifndef NDEBUG
namespace
{
bool Overlaps(std::span<const float> input, std::span<float> output) noexcept
{
    const auto input_begin = reinterpret_cast<uintptr_t>(input.data());
    const auto input_end = input_begin + input.size_bytes();
    const auto output_begin = reinterpret_cast<uintptr_t>(output.data());
    const auto output_end = output_begin + output.size_bytes();
    return input_begin < output_end && output_begin < input_end;
}
} // namespace
#endif

namespace sfFDN
{
ChannelMatrix::ChannelMatrix(const ChannelMatrixOptions& options)
    : input_channel_count_(detail::RequireValidOptions(options).input_channel_count)
    , output_channel_count_(options.output_channel_count)
    , coefficients_(options.coefficients)
{
}

void ChannelMatrix::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.ChannelCount() == input_channel_count_);
    assert(output.ChannelCount() == output_channel_count_);
    assert(input.SampleCount() == output.SampleCount());

#ifndef NDEBUG
    for (uint32_t input_channel = 0; input_channel < input_channel_count_; ++input_channel)
    {
        const auto input_samples = input.GetChannelSpan(input_channel);
        for (uint32_t output_channel = 0; output_channel < output_channel_count_; ++output_channel)
        {
            assert(!Overlaps(input_samples, output.GetChannelSpan(output_channel)));
        }
    }
#endif

    for (uint32_t output_channel = 0; output_channel < output_channel_count_; ++output_channel)
    {
        const size_t row_offset = static_cast<size_t>(output_channel) * input_channel_count_;
        auto output_samples = output.GetChannelSpan(output_channel);
        ArrayMath::Scale(input.GetChannelSpan(0), coefficients_[row_offset], output_samples);
        for (uint32_t input_channel = 1; input_channel < input_channel_count_; ++input_channel)
        {
            ArrayMath::ScaleAccumulate(input.GetChannelSpan(input_channel), coefficients_[row_offset + input_channel],
                                       output_samples);
        }
    }
}

uint32_t ChannelMatrix::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return input_channel_count_;
}

uint32_t ChannelMatrix::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return output_channel_count_;
}

void ChannelMatrix::Clear()
{
}

std::unique_ptr<AudioProcessor> ChannelMatrix::Clone() const
{
    return std::make_unique<ChannelMatrix>(ChannelMatrixOptions{
        .input_channel_count = input_channel_count_,
        .output_channel_count = output_channel_count_,
        .coefficients = coefficients_,
    });
}

} // namespace sfFDN
