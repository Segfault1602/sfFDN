// SPDX-License-Identifier: MIT
//
// NOTICE:
// This file was created by Claude Opus 5, it appears to work and it is genuinely fast on my M3 macbook.
// Feel free to steal this code.
#pragma once

#include "simd.h"

#include "sffdn/attributes.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/types.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace sfFDN
{

/**
 * @brief A bank of identical-length biquad cascades evaluated across channels with SIMD.
 *
 * Each channel owns its own cascade of transposed-direct-form-II biquads. Because the recurrence
 * runs along time, vectorization is applied across the channel dimension: one SIMD lane per
 * channel.
 *
 * AudioBuffer stores audio channel-major, so a group of `simd::kWidth` channels is first
 * interleaved into a contiguous chunk-sized scratch buffer. All cascade stages then run over that
 * scratch, which keeps the stage coefficients and filter state resident in registers for the whole
 * chunk, before the result is written back out. Working on a fixed-size chunk bounds the scratch
 * buffer so it can be allocated once in SetCoefficients() and never on the audio thread.
 *
 * Terminology: a *group* is one SIMD vector's worth of channels, so exactly `simd::kWidth` (4)
 * channels. A *pass* evaluates several groups side by side; `kGroups` counts vectors, not lanes,
 * so a pass of 8 groups covers 8 * 4 = 32 channels held in 8 separate registers. A *chunk* is the
 * number of samples processed between state reloads.
 *
 * The arithmetic matches the scalar CascadedBiquads kernel stage for stage:
 *     y  = b0 * x + s0
 *     s0 = b1 * x + s1 - a1 * y
 *     s1 = b2 * x      - a2 * y
 */
class SimdBiquadBank
{
  public:
    SimdBiquadBank() = default;

    /**
     * @brief Configures the bank.
     * @param coeffs Cascade coefficients indexed as `coeffs[channel * stage_count + stage]`.
     * @param channel_count Number of channels; every channel must have the same number of stages.
     */
    void SetCoefficients(std::span<const FilterCoefficients> coeffs, uint32_t channel_count)
    {
        if (channel_count == 0 || coeffs.size() % channel_count != 0)
        {
            throw std::runtime_error("Invalid coefficient size");
        }

        channel_count_ = channel_count;
        stage_count_ = static_cast<uint32_t>(coeffs.size() / channel_count);
        lane_count_ = static_cast<uint32_t>(simd::PadToWidth(channel_count));

        const size_t total = static_cast<size_t>(stage_count_) * lane_count_;

        // Padding lanes keep zero coefficients and zero state so they can never produce
        // denormals, infinities, or NaNs that would slow down the active lanes.
        b0_.assign(total, 0.f);
        b1_.assign(total, 0.f);
        b2_.assign(total, 0.f);
        a1_.assign(total, 0.f);
        a2_.assign(total, 0.f);
        s0_.assign(total, 0.f);
        s1_.assign(total, 0.f);

        for (uint32_t channel = 0; channel < channel_count_; ++channel)
        {
            for (uint32_t stage = 0; stage < stage_count_; ++stage)
            {
                const FilterCoefficients normalized = coeffs[(channel * stage_count_) + stage].Normalize();
                const size_t index = (static_cast<size_t>(stage) * lane_count_) + channel;
                b0_[index] = normalized.b0;
                b1_[index] = normalized.b1;
                b2_[index] = normalized.b2;
                a1_[index] = normalized.a1;
                a2_[index] = normalized.a2;
            }
        }

        // Zeroed once here; inactive lanes of the trailing group are never written afterwards.
        scratch_.assign(static_cast<size_t>(kChunkSize) * simd::kWidth * kMaxGroups, 0.f);
    }

    void Clear()
    {
        std::ranges::fill(s0_, 0.f);
        std::ranges::fill(s1_, 0.f);
    }

    uint32_t ChannelCount() const noexcept SFFDN_NONBLOCKING
    {
        return channel_count_;
    }

    uint32_t StageCount() const noexcept SFFDN_NONBLOCKING
    {
        return stage_count_;
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
    {
        assert(input.SampleCount() == output.SampleCount());
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == channel_count_);
        assert(!scratch_.empty());

        const uint32_t sample_count = input.SampleCount();
        constexpr auto kWidth = static_cast<uint32_t>(simd::kWidth);

        // When a split is needed the groups are spread evenly rather than packing passes full and
        // leaving a small remainder: a trailing one-group pass runs at the latency of a single
        // dependency chain and can cost as much as a full pass.
        const uint32_t total_groups = (channel_count_ + kWidth - 1) / kWidth;
        uint32_t groups_per_pass = total_groups;
        if (total_groups > kMaxGroups)
        {
            const uint32_t pass_count = (total_groups + kPreferredGroups - 1) / kPreferredGroups;
            groups_per_pass = (total_groups + pass_count - 1) / pass_count;
        }

        for (uint32_t first_group = 0; first_group < total_groups; first_group += groups_per_pass)
        {
            const uint32_t groups = std::min(groups_per_pass, total_groups - first_group);
            const uint32_t base = first_group * kWidth;
            const uint32_t channels = std::min(groups * kWidth, channel_count_ - base);
            const uint32_t row_stride = groups * kWidth;

            for (uint32_t offset = 0; offset < sample_count; offset += kChunkSize)
            {
                const uint32_t chunk = std::min(kChunkSize, sample_count - offset);

                Interleave(input, base, channels, row_stride, offset, chunk);

                // The biquad recurrence is serial in time, so a single group leaves the FMA
                // pipeline mostly idle waiting on its own dependency chain. Running several
                // groups' chains side by side inside one stage fills those latency slots.
                switch (groups)
                {
                case 8:
                    RunStages<8>(base, chunk);
                    break;
                case 7:
                    RunStages<7>(base, chunk);
                    break;
                case 6:
                    RunStages<6>(base, chunk);
                    break;
                case 5:
                    RunStages<5>(base, chunk);
                    break;
                case 4:
                    RunStages<4>(base, chunk);
                    break;
                case 3:
                    RunStages<3>(base, chunk);
                    break;
                case 2:
                    RunStages<2>(base, chunk);
                    break;
                default:
                    RunStages<1>(base, chunk);
                    break;
                }

                Deinterleave(output, base, channels, row_stride, offset, chunk);
            }
        }
    }

  private:
    // Chosen so the scratch buffer stays comfortably inside L1 while still amortizing the
    // per-stage state load/store across many samples.
    static constexpr uint32_t kChunkSize = 64;

    // Independent biquad chains evaluated together to hide FMA latency. Measured cost per group
    // on an M3 falls steeply as chains are added (1 group 2.06 us, 2 groups 1.33, 4 groups 0.93,
    // 6 groups 0.75, 7 groups 0.72) and then rises again at 8 groups (0.87) as the live state
    // outgrows the register file.
    //
    // Up to kMaxGroups groups still run as a single pass, because avoiding a second pass entirely
    // beats the slightly worse per-group rate. Once a split is unavoidable, passes are sized at
    // kPreferredGroups, which is the most efficient width measured.
    static constexpr uint32_t kMaxGroups = 8;
    static constexpr uint32_t kPreferredGroups = 6;

    /**
     * @brief Gathers a pass's channels from the caller's channel-major buffer into the frame-major
     * scratch layout the SIMD kernel reads.
     *
     * AudioBuffer keeps each channel contiguous, so one sample of four channels is spread across
     * four distant addresses. The kernel instead needs those four values adjacent, so it can load
     * them as a single vector. This transposes one chunk into that form:
     *
     * @verbatim
     * input (channel-major)        scratch (frame-major, row_stride floats per row)
     *   ch0: a0 a1 a2 ...            row 0: a0 b0 c0 d0 | e0 f0 g0 h0 | ...
     *   ch1: b0 b1 b2 ...            row 1: a1 b1 c1 d1 | e1 f1 g1 h1 | ...
     *   ch2: c0 c1 c2 ...            row 2: a2 b2 c2 d2 | e2 f2 g2 h2 | ...
     *   ...                                 <- group 0 -> <- group 1 ->
     * @endverbatim
     *
     * Each row holds one frame, so `RunStages()` loads group `g` of row `t` as one contiguous
     * vector at `scratch_[t * row_stride + g * simd::kWidth]`.
     *
     * The transpose is done once per chunk rather than per stage, so its cost is amortized over
     * every cascade stage.
     *
     * @param input Source buffer, channel-major.
     * @param base Index of this pass's first channel.
     * @param channels Real channels in this pass; may be less than @p row_stride on the last pass.
     * @param row_stride Floats per scratch row, equal to `groups * simd::kWidth`.
     * @param offset First sample of this chunk within the block.
     * @param chunk Samples in this chunk.
     */
    void Interleave(const AudioBuffer& input, uint32_t base, uint32_t channels, uint32_t row_stride, uint32_t offset,
                    uint32_t chunk) noexcept SFFDN_NONBLOCKING
    {
        const std::span<float> rows = std::span<float>(scratch_).first(static_cast<size_t>(chunk) * row_stride);

        for (uint32_t lane = 0; lane < channels; ++lane)
        {
            const std::span<const float> source = input.GetChannelSpan(base + lane).subspan(offset, chunk);
            for (uint32_t sample = 0; sample < chunk; ++sample)
            {
                rows[(static_cast<size_t>(sample) * row_stride) + lane] = source[sample];
            }
        }

        // Padding lanes carry zero coefficients, so their output is zero regardless of input.
        // They are still cleared here so a stale value from a previous pass can never combine
        // with a non-finite sample to produce a NaN that would poison the active lanes.
        for (uint32_t lane = channels; lane < row_stride; ++lane)
        {
            for (uint32_t sample = 0; sample < chunk; ++sample)
            {
                rows[(static_cast<size_t>(sample) * row_stride) + lane] = 0.f;
            }
        }
    }

    /**
     * @brief Scatters a filtered chunk from the frame-major scratch back to the channel-major
     * output buffer. The exact inverse of Interleave().
     *
     * Only the @p channels real lanes are written back; padding lanes are discarded. Reading and
     * writing a separate output buffer is what allows Process() to support `input == output`
     * safely, because a chunk is fully consumed from scratch before it is overwritten.
     *
     * @param output Destination buffer, channel-major.
     * @param base Index of this pass's first channel.
     * @param channels Real channels in this pass; padding lanes above this are not written.
     * @param row_stride Floats per scratch row, equal to `groups * simd::kWidth`.
     * @param offset First sample of this chunk within the block.
     * @param chunk Samples in this chunk.
     */
    void Deinterleave(AudioBuffer& output, uint32_t base, uint32_t channels, uint32_t row_stride, uint32_t offset,
                      uint32_t chunk) noexcept SFFDN_NONBLOCKING
    {
        const std::span<const float> rows =
            std::span<const float>(scratch_).first(static_cast<size_t>(chunk) * row_stride);

        for (uint32_t lane = 0; lane < channels; ++lane)
        {
            const std::span<float> destination = output.GetChannelSpan(base + lane).subspan(offset, chunk);
            for (uint32_t sample = 0; sample < chunk; ++sample)
            {
                destination[sample] = rows[(static_cast<size_t>(sample) * row_stride) + lane];
            }
        }
    }

    /**
     * @brief Runs the full biquad cascade in place over one interleaved chunk of scratch.
     *
     * Loops stages on the outside and samples on the inside. That order lets one stage's five
     * coefficient vectors and two state vectors per group be loaded once and then stay in
     * registers for the entire chunk, so the inner loop is pure arithmetic on contiguous data:
     * one vector load, four fused multiply-adds and one multiply, then one vector store, per
     * group per sample.
     *
     * @p kGroups is a compile-time parameter so the inner group loop unrolls fully and the state
     * arrays become registers rather than memory. Process() dispatches on the pass's group count
     * through a switch, which is why every instantiation from 1 to kMaxGroups exists.
     *
     * Multiple groups are evaluated together purely to expose instruction-level parallelism, not
     * to widen the vectors. A biquad is serial in time (`s0` feeds the next sample's `y`, which
     * feeds the next `s0`), so a single group stalls on its own FMA latency chain. Separate
     * groups are independent channels, so their chains interleave and fill those stalls. See the
     * kMaxGroups comment for the measured cost curve.
     *
     * Per group and stage the recurrence matches the scalar CascadedBiquads kernel exactly, which
     * is why the two produce bit-identical output:
     *     y  = b0 * x + s0
     *     s0 = b1 * x + s1 - a1 * y
     *     s1 = b2 * x      - a2 * y
     *
     * Padding lanes hold zero coefficients and zero state, so they compute zeros and cannot
     * disturb the real lanes.
     *
     * State is written back to s0_ and s1_ at the end of each stage so it carries across chunks
     * and across Process() calls, which keeps the filter continuous over block boundaries.
     *
     * @tparam kGroups Vectors evaluated side by side; covers `kGroups * simd::kWidth` channels.
     * @param base Index of this pass's first channel; selects the coefficient and state slice.
     * @param chunk Samples in this chunk, at most kChunkSize.
     */
    template <uint32_t kGroups>
    void RunStages(uint32_t base, uint32_t chunk) noexcept SFFDN_NONBLOCKING
    {
        constexpr size_t kRowStride = kGroups * simd::kWidth;

        const std::span<const float> b0_lanes{b0_};
        const std::span<const float> b1_lanes{b1_};
        const std::span<const float> b2_lanes{b2_};
        const std::span<const float> a1_lanes{a1_};
        const std::span<const float> a2_lanes{a2_};
        const std::span<float> s0_lanes{s0_};
        const std::span<float> s1_lanes{s1_};

        // Only the rows this chunk actually occupies; every row below is a subspan of this.
        const std::span<float> rows = std::span<float>(scratch_).first(static_cast<size_t>(chunk) * kRowStride);

        for (uint32_t stage = 0; stage < stage_count_; ++stage)
        {
            const size_t index = (static_cast<size_t>(stage) * lane_count_) + base;

            std::array<simd::Vec, kGroups> b0{};
            std::array<simd::Vec, kGroups> b1{};
            std::array<simd::Vec, kGroups> b2{};
            std::array<simd::Vec, kGroups> a1{};
            std::array<simd::Vec, kGroups> a2{};
            std::array<simd::Vec, kGroups> s0{};
            std::array<simd::Vec, kGroups> s1{};

            for (uint32_t g = 0; g < kGroups; ++g)
            {
                const size_t offset = index + (g * simd::kWidth);
                b0[g] = simd::Load(simd::LanesAt(b0_lanes, offset));
                b1[g] = simd::Load(simd::LanesAt(b1_lanes, offset));
                b2[g] = simd::Load(simd::LanesAt(b2_lanes, offset));
                a1[g] = simd::Load(simd::LanesAt(a1_lanes, offset));
                a2[g] = simd::Load(simd::LanesAt(a2_lanes, offset));
                s0[g] = simd::Load(simd::LanesAt(s0_lanes, offset));
                s1[g] = simd::Load(simd::LanesAt(s1_lanes, offset));
            }

            for (uint32_t sample = 0; sample < chunk; ++sample)
            {
                const std::span<float> row = rows.subspan(static_cast<size_t>(sample) * kRowStride, kRowStride);

                for (uint32_t g = 0; g < kGroups; ++g)
                {
                    const std::span<float, simd::kWidth> slot = simd::LanesAt(row, g * simd::kWidth);

                    const simd::Vec x = simd::Load(slot);
                    const simd::Vec y = simd::MulAdd(b0[g], x, s0[g]);

                    s0[g] = simd::NegMulAdd(a1[g], y, simd::MulAdd(b1[g], x, s1[g]));
                    s1[g] = simd::NegMulAdd(a2[g], y, simd::Mul(b2[g], x));

                    simd::Store(slot, y);
                }
            }

            for (uint32_t g = 0; g < kGroups; ++g)
            {
                const size_t offset = index + (g * simd::kWidth);
                simd::Store(simd::LanesAt(s0_lanes, offset), s0[g]);
                simd::Store(simd::LanesAt(s1_lanes, offset), s1[g]);
            }
        }
    }

    uint32_t channel_count_{0};
    uint32_t stage_count_{0};
    uint32_t lane_count_{0};

    std::vector<float> b0_;
    std::vector<float> b1_;
    std::vector<float> b2_;
    std::vector<float> a1_;
    std::vector<float> a2_;

    std::vector<float> s0_;
    std::vector<float> s1_;

    std::vector<float> scratch_;
};

} // namespace sfFDN
