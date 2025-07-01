#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "audio_processor.h"
#include "filter.h"

namespace sfFDN
{
class FilterBank : public AudioProcessor
{
  public:
    FilterBank();
    ~FilterBank();

    void Clear();

    void AddFilter(std::unique_ptr<AudioProcessor> filter);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

  private:
    std::vector<AudioProcessor*> filters_;
};

} // namespace sfFDN