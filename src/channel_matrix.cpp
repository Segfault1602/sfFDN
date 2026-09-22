#include "sffdn/channel_matrix.h"

#include "array_math.h"
#include "audio_buffer_alias.h"
#include "processor_option_validation.h"
#include "sffdn/audio_buffer.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>

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
    assert(ClassifyAudioBufferAlias(input, output) == AudioBufferAlias::Disjoint);
#endif

    for (uint32_t output_channel = 0; output_channel < output_channel_count_; ++output_channel)
    {
        const size_t row_offset = static_cast<size_t>(output_channel) * input_channel_count_;
        const auto output_samples = output.GetChannelSpan(output_channel);
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
