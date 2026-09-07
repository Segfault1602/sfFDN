// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "attributes.h"
#include "audio_buffer.h"
#include "audio_processor.h"
#include "delaybank.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace sfFDN
{

/** @brief The immutable shape of an FDN: M external inputs, N delay lines, K external outputs.
 *
 * All four counts are fixed for the lifetime of the FDN. Every processor installed afterwards is validated against
 * them, so a topology cannot change out from under an already-configured network.
 */
struct FDNTopology
{
    /** The number of delay lines, N. Must be greater than zero. */
    uint32_t order = 0;
    /** The size of the audio blocks processed in the main loop. Must be at least 1. */
    uint32_t block_size = 0;
    /** The number of external input channels, M. Must be greater than zero. */
    uint32_t input_channel_count = 1;
    /** The number of external output channels, K. Must be greater than zero. */
    uint32_t output_channel_count = 1;
    /** Whether to use the transposed topology. */
    bool transposed = false;
};

/** FDN (Feedback Delay Network) class */
class FDN : public AudioProcessor
{
  public:
    /** @brief Constructs an FDN with the given immutable topology.
     * @param topology The M/N/K channel counts, block size, and topology arrangement.
     * @throws std::invalid_argument if any of the channel counts is zero or the block size is smaller than 1.
     *
     * `block_size` is used to allocate internal buffers for processing. The order and the external channel counts
     * cannot be changed afterwards; construct a new FDN instead.
     */
    explicit FDN(const FDNTopology& topology);

    /** @brief Constructs a one-input, one-output FDN with a specified order (number of delay lines).
     * @param order The number of delay lines. Must be greater than zero.
     * @param block_size The size of the audio blocks to be processed in the main loop.
     * @param transpose Whether to use transposed configuration.
     *
     * Convenience overload equivalent to `FDN(FDNTopology{.order = order, .block_size = block_size, .transposed =
     * transpose})`.
     */
    FDN(uint32_t order, uint32_t block_size, bool transpose = false);

    ~FDN() override = default;

    FDN(const FDN&) = delete;
    FDN& operator=(const FDN&) = delete;

    /** @brief Move constructor for the FDN.
     */
    FDN(FDN&& other) noexcept;

    /** @brief Move assignment operator for the FDN.
     * @return A reference to the assigned FDN.
     */
    FDN& operator=(FDN&& other) noexcept;

    /** @brief Get the FDN's order (number of delay lines, N).
     * @returns The number of delay lines. Fixed at construction.
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
     * @brief Set the input boundary processor B.
     *
     * @param gains The processor to use for input routing. Its input channel count must equal InputChannelCount() and
     * its output channel count must equal GetOrder(). The FDN takes ownership of the pointer.
     * @return true if the gains were set successfully
     * @return false if the gains could not be set
     *
     * False is returned if the pointer is null or the channel counts do not match the fixed topology.
     */
    bool SetInputGains(std::unique_ptr<AudioProcessor> gains);

    /** @brief Set the output boundary processor C.
     * @param gains The processor to use for output routing. Its input channel count must equal GetOrder() and its
     * output channel count must equal OutputChannelCount(). The FDN takes ownership of the pointer.
     * @return true if the gains were set successfully
     * @return false if the gains could not be set
     *
     * False is returned if the pointer is null or the channel counts do not match the fixed topology.
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

    /** @brief Set a direct-path processor mapping the M input channels to the K output channels.
     * @param direct The processor to use for the direct (dry) path, or nullptr to remove it. The FDN takes ownership
     * of the pointer.
     * @return true if the direct path was set or removed successfully, false if its channel counts do not match
     * InputChannelCount() and OutputChannelCount().
     *
     * Removing the direct processor falls back to the scalar direct gain when M equals K, and to a silent direct path
     * otherwise.
     */
    bool SetDirectPath(std::unique_ptr<AudioProcessor> direct);

    /** @brief Get the direct-path processor, or nullptr when the scalar direct gain is active. */
    AudioProcessor* GetDirectPath() const;

    /** @brief Set the scalar direct gain, applied as a diagonal `gain * I` direct path.
     *
     * Only valid when InputChannelCount() equals OutputChannelCount(); the call is otherwise ignored. Setting a
     * scalar gain removes any direct-path processor installed with SetDirectPath().
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
    bool SetLoopFilter(std::unique_ptr<AudioProcessor> filter_bank);

    /** @brief Get the Filter Bank AudioProcessor.
     * @returns A pointer to the Filter Bank AudioProcessor, or nullptr if not set.
     */
    AudioProcessor* GetLoopFilter() const;

    /** @brief Set the delay bank.
     * @param config The configuration for the delay bank.
     * @return true if the delay bank was set successfully
     * @return false if the delay bank could not be set.
     */
    bool SetDelayBank(const DelayBankOptions& config);

    /** @brief Set the delays.
     * @param delays A span of delay lengths in samples. The size of the span must be equal to GetOrder().
     * @param interpolation_type The type of interpolation to use for the delays.
     * @return true if the delays were set successfully
     * @return false if the delays could not be set. Happens if the size of the span does not match GetOrder() or if any
     * of the delay lengths are smaller than the block_size set in the constructor.
     */
    bool SetDelays(std::span<const float> delays,
                   DelayInterpolationType interpolation_type = DelayInterpolationType::None);

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
     * @return false if the filter could not be set. Only happens if its input or output channel count does not equal
     * OutputChannelCount().
     */
    bool SetTCFilter(std::unique_ptr<AudioProcessor> filter);

    /** @brief Get the Tone Correction Filter AudioProcessor.
     * @returns A pointer to the Tone Correction Filter AudioProcessor, or nullptr if not set.
     */
    AudioProcessor* GetTCFilter() const;

    /** @brief Process audio buffers and accumulate the result into output.
     * @param input The input audio buffer. Its channel count must equal InputChannelCount().
     * @param output The output audio buffer. Its channel count must equal OutputChannelCount(). A mono FDN also
     * accepts additional output channels and duplicates channel zero for backward compatibility.
     *
     * The input and output buffers must have the same sample count.
     * input.SampleCount() does not have to be equal to block_size but it is recommended for optimal performance.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels this processor expects. */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels this processor produces. */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

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
    void PrepareOutput(const AudioBuffer& input, const AudioBuffer& wet_input) noexcept SFFDN_NONBLOCKING;
    void AccumulateOutput(AudioBuffer& output) noexcept SFFDN_NONBLOCKING;
    void TickInternal(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING;
    void Tick(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING;
    void TickTranspose(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING;
    void TickTransposeInternal(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING;

    DelayBank delay_bank_;
    std::unique_ptr<AudioProcessor> filter_bank_;
    std::unique_ptr<AudioProcessor> mixing_matrix_;

    std::unique_ptr<AudioProcessor> input_gains_;
    std::unique_ptr<AudioProcessor> output_gains_;
    std::unique_ptr<AudioProcessor> direct_path_;

    uint32_t order_;
    uint32_t block_size_;
    uint32_t input_channel_count_;
    uint32_t output_channel_count_;
    float direct_gain_;

    std::vector<float> feedback_;
    std::vector<float> temp_buffer_;
    std::vector<float> wet_output_;
    std::vector<float> tone_output_;
    std::vector<float> direct_output_;

    std::unique_ptr<AudioProcessor> tc_filter_;

    bool transpose_;
};
} // namespace sfFDN