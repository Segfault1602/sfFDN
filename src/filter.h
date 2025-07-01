#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "audio_processor.h"

namespace sfFDN
{
class Filter
{
    static constexpr uint32_t COEFFICIENT_COUNT = 3;

  public:
    Filter() = default;
    virtual ~Filter() = default;

    virtual void Clear();

    /// @brief  Tick the filter.
    /// @param in Input sample
    /// @return Output sample
    virtual float Tick(float in) = 0;

    /// @brief Filter a block of samples. 'in' and 'out' can be the same buffer.
    /// @param in The input buffer.
    /// @param out The output buffer.
    /// @param size The size of the buffer.
    virtual void ProcessBlock(const float* in, float* out, size_t size);

    /// @brief Set the gain of the filter.
    /// @param gain
    void SetGain(float gain);

    /// @brief Set the 'a' coefficients of the filter.
    /// @param a Array of size COEFFICIENT_COUNT containing the 'a' coefficients.
    void SetA(const float (&a)[COEFFICIENT_COUNT]);

    /// @brief Set the 'b' coefficients of the filter.
    /// @param b Array of size COEFFICIENT_COUNT containing the 'b' coefficients.
    void SetB(const float (&b)[COEFFICIENT_COUNT]);

  protected:
    /// @brief The gain applied to the input of the filter.
    float gain_ = 1.f;

    /// @brief The 'b' coefficients of the filter.
    std::array<float, 3> b_ = {0.f, 0.f, 0.f};

    /// @brief The 'a' coefficients of the filter.
    std::array<float, 3> a_ = {1.f, 0.f, 0.f};

    /// @brief The previous outputs of the filter.
    std::array<float, 3> outputs_ = {0.f};
    /// @brief The previous inputs of the filter.
    std::array<float, 3> inputs_ = {0.f};
};

/// @brief Implements a simple one pole filter with differential equation y(n) = b0*x(n) - a1*y(n-1)
class OnePoleFilter : public Filter, public AudioProcessor
{
  public:
    OnePoleFilter() = default;
    ~OnePoleFilter() override = default;

    /// @brief Set the pole of the filter.
    /// @param pole The pole of the filter.
    void SetPole(float pole);

    void SetCoefficients(float b0, float a1);

    /// @brief Set the pole of the filter to obtain an exponential decay filter.
    /// @param decayDb The decay in decibels.
    /// @param timeMs The time in milliseconds.
    /// @param samplerate The samplerate.
    void SetDecayFilter(float decayDb, float timeMs, float samplerate);

    /// @brief Set the pole of the filter to obtain a lowpass filter with a 3dB cutoff frequency.
    /// @param cutoff The cutoff frequency, normalized between 0 and 1.
    void SetLowpass(float cutoff);

    /// @brief Input a sample in the filter and return the next output
    /// @param in The input sample
    /// @return The next output sample
    float Tick(float in) override;

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;
};

class CascadedBiquads : public Filter, public AudioProcessor
{
  public:
    CascadedBiquads();
    ~CascadedBiquads() override;

    void Clear() override;

    void SetCoefficients(uint32_t num_stage, std::span<const float> coeffs);

    /// @brief Input a sample in the filter and return the next output
    /// @param in The input sample
    /// @return The next output sample
    float Tick(float in) override;

    void ProcessBlock(const float* in, float* out, size_t size) override;

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

    void dump_coeffs();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace sfFDN