#pragma once

#include "sffdn/audio_buffer.h"

namespace sfFDN
{
/** @brief Base class for oscillators and signal generators. */
class Generator
{
  public:
    Generator() = default;
    virtual ~Generator() = default;

    /** @brief Fills the output span with generated samples. */
    virtual void Generate(std::span<float> output) = 0;
};

/** @brief A sine wave oscillator. */
class SineWave : public Generator
{
  public:
    /** @brief Constructs a sine wave oscillator.
     * @param frequency The frequency of the sine wave in, normalized [0, 1].
     * @param initial_phase The initial phase of the sine wave, normalized [0, 1].
     */
    SineWave(float frequency = 0.0f, float initial_phase = 0.0f);

    /** @brief Resets the phase of the sine wave oscillator. */
    void ResetPhase();

    /** @brief Sets the frequency of the sine wave oscillator.
     * @param frequency The frequency of the sine wave in, normalized [0, 1].
     */
    void SetFrequency(float frequency);

    /** @brief Sets the amplitude of the sine wave oscillator.
     * @param amplitude The amplitude of the sine wave.
     */
    void SetAmplitude(float amplitude);

    /** @brief Sets the DC offset of the sine wave oscillator.
     * @param offset The offset of the sine wave.
     */
    void SetOffset(float offset);

    /** @brief Returns the amplitude of the sine wave oscillator. */
    float GetAmplitude() const;

    /** @brief Returns the DC offset of the sine wave oscillator. */
    float GetOffset() const;

    /** @brief Returns the next output sample without advancing the phase. */
    float NextOut() const;

    /** @brief Advances the phase and returns the next output sample. */
    float Tick();

    /** @brief Fills the output span with generated samples. */
    void Generate(std::span<float> output) override;

    /**
     * @brief Multiply `input` by the sine wave and store the result in `output`.
     * @param input The input signal to modulate.
     * @param output The output signal to store the result.
     */
    void Multiply(std::span<const float> input, std::span<float> output);

    /**
     * @brief Multiply `input` by the sine wave and accumulate the result in `output`.
     * @param input The input signal to modulate.
     * @param output The output signal to store the result.
     */
    void MultiplyAccumulate(std::span<const float> input, std::span<float> output);

  private:
    float phase_;
    float phase_increment_;

    float amplitude_;
    float offset_;
    float phase_offset_;
};

} // namespace sfFDN