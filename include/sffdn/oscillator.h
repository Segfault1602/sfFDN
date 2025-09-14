#pragma once

#include "sffdn/audio_buffer.h"

namespace sfFDN
{
class Generator
{
  public:
    Generator() = default;
    virtual ~Generator() = default;

    virtual void Generate(std::span<float> output) = 0;
};

class SineWave : public Generator
{
  public:
    SineWave(float frequency = 0.0f, float initial_phase = 0.0f);

    void ResetPhase();

    void SetFrequency(float frequency);
    void SetAmplitude(float amplitude);
    void SetOffset(float offset);

    float GetAmplitude() const;
    float GetOffset() const;

    float NextOut() const;

    float Tick();

    void Generate(std::span<float> output) override;

    /// @brief Multiply `input` by the sine wave and store the result in `output`.
    /// @param input The input signal to modulate.
    /// @param output The output signal to store the result.
    void Multiply(std::span<const float> input, std::span<float> output);

    /// @brief Multiply `input` by the sine wave and accumulate the result in `output`.
    /// @param input The input signal to modulate.
    /// @param output The output signal to store the result.
    void MultiplyAccumulate(std::span<const float> input, std::span<float> output);

  private:
    float phase_;
    float phase_increment_;

    float amplitude_;
    float offset_;
    float phase_offset_;
};

} // namespace sfFDN