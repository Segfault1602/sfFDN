#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "sffdn/sffdn.h"

std::unique_ptr<sfFDN::FilterFeedbackMatrix> CreateFFM(uint32_t mat_size, uint32_t stage_count, uint32_t sparsity);

std::unique_ptr<sfFDN::AudioProcessor> GetFilterBank(uint32_t channel_count, uint32_t order);

std::unique_ptr<sfFDN::ParallelGains> GetDefaultInputGains(uint32_t count);
std::unique_ptr<sfFDN::ParallelGains> GetDefaultOutputGains(uint32_t count);
std::vector<uint32_t> GetDefaultDelays(uint32_t count);
std::unique_ptr<sfFDN::AudioProcessor> GetDefaultTCFilter();

std::unique_ptr<sfFDN::FDN> CreateFDN(uint32_t block_size, uint32_t fdn_order);

std::vector<float> ReadWavFile(const std::string& filename);
void WriteWavFile(const std::string& filename, const std::vector<float>& data);

std::vector<float> GetImpulseResponse(sfFDN::AudioProcessor* filter);