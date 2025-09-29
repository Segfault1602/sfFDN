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

// From: https://github.com/jatinchowdhury18/FIRBenchmarks/blob/master/src/InnerProdFIR.h
struct InnerProdFIR
{
  public:
    InnerProdFIR(std::vector<float> fir)
        : order(fir.size())
    {
        // allocate memory
        // (smart pointers would be preferred, but introduce a small overhead)
        h = std::vector<float>(order);
        z = std::vector<float>(2 * order);

        std::ranges::fill(z, 0.0f);        // clear existing state
        std::ranges::copy(fir, h.begin()); // copy FIR coefficients
    }

    void Process(sfFDN::AudioBuffer& b)
    {
        auto buffer = b.GetChannelSpan(0);
        const int numSamples = b.SampleCount();

        float y = 0.0f;
        for (int n = 0; n < numSamples; ++n)
        {
            // insert input into double-buffered state
            z[zPtr] = buffer[n];
            z[zPtr + order] = buffer[n];

            auto z_span = std::span(z).subspan(zPtr, order);

            // compute inner product over kernel and double-buffer state
            y = std::inner_product(z_span.begin(), z_span.end(), h.begin(), 0.0f);

            zPtr = (zPtr == 0 ? order - 1 : zPtr - 1); // iterate state pointer in reverse

            buffer[n] = y;
        }
    }

  private:
    const int order;
    int zPtr = 0;         // state pointer
    std::vector<float> z; // filter state
    std::vector<float> h; // filter kernel
};