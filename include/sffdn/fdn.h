// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "audio_buffer.h"
#include "audio_processor.h"
#include "delaybank.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace sfFDN
{

/** FDN (Feedback Delay Network) class */
class FDN : public AudioProcessor
{
  public:
    /** @brief Constructs an FDN with a specified order (number of channels).
     * @param order The number of channels. Must be at least 4.
     * @param block_size The size of the audio blocks to be processed in the main loop.
     * @param transpose Whether to use transposed configuration.
     *
     * `block_size` is used to allocate internal buffers for processing.
     */
    FDN(uint32_t order, uint32_t block_size = 0, bool transpose = false);

    ~FDN() = default;

    FDN(const FDN&) = delete;
    FDN& operator=(const FDN&) = delete;

    FDN(FDN&&) noexcept;
    FDN& operator=(FDN&&) noexcept;

    /**
     * @brief Set the number of channels of the FDN
     * @param order The number of channels. Must be at least 4.
     * Calling this method will reset the internal components of the FDN to the default state.
     */
    void SetOrder(uint32_t order);

    /** @brief Get the FDN's order (number of channels).
     * @returns The number of channels.
     */
    uint32_t GetOrder() const;

    /** @brief Set whether to use transposed configuration.
     * @param transpose true to use transposed configuration, false otherwise.
     *
     * In transposed configuration, the input signal is fed to the feedback matrix first and the output of the feedback
     * matrix is fed to the output of the FDN.
     */
    void SetTranspose(bool transpose);

    /** @brief Get whether the FDN is using transposed configuration.
     * @returns true if using transposed configuration, false otherwise.
     *
     * * In transposed configuration, the input signal is fed to the feedback matrix first and the output of the
     * feedback matrix is fed to the output of the FDN.
     */
    bool GetTranspose() const;

    /**
     * @brief Set the Input Gains AudioProcessor
     *
     * @param gains The AudioProcessor to use for input gains. Must have 1 input channel and number of output channels
     * equal to GetOrder(). The FDN takes ownership of the pointer.
     * @return true if the gains were set successfully
     * @return false if the gains could not be set
     *
     * False is returned if the input and output channel counts do not match the requirements.
     */
    bool SetInputGains(std::unique_ptr<AudioProcessor> gains);

    /** @brief Set the Output Gains AudioProcessor
     * @param gains The AudioProcessor to use for output gains. Must have number of input channels equal to
     * GetOrder() and 1 output channel. The FDN takes ownership of the pointer.
     * @return true if the gains were set successfully
     * @return false if the gains could not be set
     *
     * False is returned if the input and output channel counts do not match the requirements.
     */
    bool SetOutputGains(std::unique_ptr<AudioProcessor> gains);

    /** @brief Set the Input Gains from a span of floats.
     * @param gains A span of floats representing the gains for each channel. The size of the span must be equal to
     * GetOrder().
     * @return true if the gains were set successfully
     * @return false if the gains could not be set
     *
     * Returns false if the size of the span does not match GetOrder().
     * This is a convenience method that creates a ParallelGains processor in Split mode with the specified gains.
     */
    bool SetInputGains(std::span<const float> gains);

    /** @brief Set the Output Gains from a span of floats.
     * @param gains A span of floats representing the gains for each channel. The size of the span must be equal to
     * GetOrder().
     * @return true if the gains were set successfully
     * @return false if the gains could not be set
     *
     * Returns false if the size of the span does not match GetOrder().
     *
     * This is a convenience method that creates a ParallelGains processor in Merge mode with the specified gains.
     */
    bool SetOutputGains(std::span<const float> gains);

    /** @brief Get the Input Gains AudioProcessor
     * @returns A pointer to the Input Gains AudioProcessor, or nullptr if not set.
     */
    AudioProcessor* GetInputGains() const;

    /** @brief Get the Output Gains AudioProcessor
     * @returns A pointer to the Output Gains AudioProcessor, or nullptr if not set.
     */
    AudioProcessor* GetOutputGains() const;

    /** @brief Set the direct gain applied to the input signal when mixed to the output.
     */
    void SetDirectGain(float gain);

    /** @brief Set the Filter Bank AudioProcessor. The filter bank is applied inside the feedback loop, after the delay
     * lines. Also known as attenuation or absorption filters.
     * @param filter_bank The AudioProcessor to use as the filter bank. The FDN takes ownership of the pointer. Can be
     * nullptr to disable filtering.
     * @return true if the filter bank was set successfully
     * @return false if the filter bank could not be set. Only happens if filter_bank->InputChannelCount() or
     * filter_bank->OutputChannelCount() do not match GetOrder().
     */
    bool SetFilterBank(std::unique_ptr<AudioProcessor> filter_bank);

    /** @brief Get the Filter Bank AudioProcessor.
     * @returns A pointer to the Filter Bank AudioProcessor, or nullptr if not set.
     */
    AudioProcessor* GetFilterBank() const;

    /** @brief Set the delays.
     * @param delays A span of delay lengths in samples. The size of the span must be equal to GetOrder().
     * @return true if the delays were set successfully
     * @return false if the delays could not be set. Happens if the size of the span does not match GetOrder() or if any
     * of the delay lengths are smaller than the block_size set in the constructor.
     */
    bool SetDelays(const std::span<const uint32_t> delays);

    /** @brief Get the Delay Bank.
     * @returns A const reference to the Delay Bank.
     */
    const DelayBank& GetDelayBank() const;

    /** @brief Set the Feedback Matrix AudioProcessor.
     * @param mixing_matrix The AudioProcessor to use as the feedback matrix. The FDN takes ownership of the pointer.
     * Can be nullptr to disable mixing.
     * @return true if the mixing matrix was set successfully
     * @return false if the mixing matrix could not be set. Only happens if mixing_matrix->InputChannelCount() or
     * mixing_matrix->OutputChannelCount() do not match GetOrder().
     */
    bool SetFeedbackMatrix(std::unique_ptr<AudioProcessor> mixing_matrix);

    /** @brief Get the Feedback Matrix AudioProcessor.
     * @returns A pointer to the Feedback Matrix AudioProcessor, or nullptr if not set.
     */
    AudioProcessor* GetFeedbackMatrix() const;

    /** @brief Set the Tone Correction Filter AudioProcessor. The Tone correction filter is applied to the output of
     * the FDN, after the output gains.
     * @param filter The AudioProcessor to use as the tone correction filter. The FDN takes ownership of the pointer.
     * Can be nullptr to disable filtering.
     * @return true if the filter was set successfully
     * @return false if the filter could not be set. Only happens if filter->InputChannelCount() or
     * filter->OutputChannelCount() do not equal 1.
     */
    bool SetTCFilter(std::unique_ptr<AudioProcessor> filter);

    /** @brief Get the Tone Correction Filter AudioProcessor.
     * @returns A pointer to the Tone Correction Filter AudioProcessor, or nullptr if not set.
     */
    AudioProcessor* GetTCFilter() const;

    /** @brief Process audio buffers.
     * @param input The input audio buffer. Must be mono (1 channel).
     * @param output The output audio buffer. Must be mono (1 channel).
     *
     * The input and output buffers must have the same sample count.
     * input.SampleCount() does not have to be equal to block_size but it is recommended for optimal performance.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Returns the number of input channels this processor expects.
     * @return 1
     */
    uint32_t InputChannelCount() const override
    {
        return 1;
    }

    /** @brief Returns the number of output channels this processor produces.
     * @return 1
     */
    uint32_t OutputChannelCount() const override
    {
        return 1;
    }

    /** @brief Clears the internal state of the FDN.
     * This function clears the internal state of all delay banks, filter banks, and feedback matrices.
     */
    void Clear() override;

    /** @brief Creates a copy of the FDN.
     * @return A unique pointer to the cloned FDN.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

    /** @brief Creates a copy of the FDN.
     * @return A unique pointer to the cloned FDN.
     */
    std::unique_ptr<FDN> CloneFDN() const;

  private:
    void TickInternal(const AudioBuffer& input, AudioBuffer& output);
    void Tick(const AudioBuffer& input, AudioBuffer& output);
    void TickTranspose(const AudioBuffer& input, AudioBuffer& output);
    void TickTransposeInternal(const AudioBuffer& input, AudioBuffer& output);

    DelayBank delay_bank_;
    std::unique_ptr<AudioProcessor> filter_bank_;
    std::unique_ptr<AudioProcessor> mixing_matrix_;

    std::unique_ptr<AudioProcessor> input_gains_;
    std::unique_ptr<AudioProcessor> output_gains_;

    uint32_t order_;
    uint32_t block_size_;
    float direct_gain_;

    std::vector<float> feedback_;
    std::vector<float> temp_buffer_;

    std::unique_ptr<AudioProcessor> tc_filter_;

    bool transpose_;
};
} // namespace sfFDN