#include "sffdn/oscillator.h"

#include "array_math.h"
#include "json_helper.h"
#include "simd.h"
#include "sine_table.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#ifdef SFFDN_USE_VDSP
#include <Accelerate/Accelerate.h>
#endif

namespace
{
float Sine(float phase) noexcept SFFDN_NONBLOCKING
{
    assert(phase >= 0.f);

    phase = phase - std::floor(phase);

    const float index = phase * sfFDN::kSineTableSize;

    const auto uindex = static_cast<int32_t>(index);
    const auto frac = index - static_cast<float>(uindex);

    const float a = sfFDN::kSineTable[uindex];
    const float b = sfFDN::kSineTable[uindex + 1];
    return a + ((b - a) * frac);
}

} // namespace

namespace sfFDN
{
namespace
{
#ifdef SFFDN_SIMD_NEON
/** @brief Four linearly interpolated sine-table lookups. */
float32x4_t Sine4(float32x4_t phase) noexcept SFFDN_NONBLOCKING
{
    const float32x4_t wrapped = vsubq_f32(phase, vrndmq_f32(phase));
    const float32x4_t index = vmulq_n_f32(wrapped, static_cast<float>(kSineTableSize));

    const int32x4_t uindex = vcvtq_s32_f32(index);
    const float32x4_t frac = vsubq_f32(index, vcvtq_f32_s32(uindex));

    const float32x2_t pair0 = vld1_f32(&kSineTable[vgetq_lane_s32(uindex, 0)]);
    const float32x2_t pair1 = vld1_f32(&kSineTable[vgetq_lane_s32(uindex, 1)]);
    const float32x2_t pair2 = vld1_f32(&kSineTable[vgetq_lane_s32(uindex, 2)]);
    const float32x2_t pair3 = vld1_f32(&kSineTable[vgetq_lane_s32(uindex, 3)]);

    // De-interleave the four adjacent table-entry pairs into lower and upper vectors.
    const float32x4_t low = vcombine_f32(pair0, pair1);
    const float32x4_t high = vcombine_f32(pair2, pair3);
    const float32x4_t a = vuzp1q_f32(low, high);
    const float32x4_t b = vuzp2q_f32(low, high);

    return vfmaq_f32(a, vsubq_f32(b, a), frac);
}
#endif

template <typename VectorSink, typename ScalarSink>
float RunOscillator(size_t count, float phase, float increment, const std::array<float, 3>& wave,
                    [[maybe_unused]] VectorSink vector_sink, ScalarSink scalar_sink) noexcept SFFDN_NONBLOCKING
{
    const auto [phase_offset, amplitude, offset] = wave;

    constexpr size_t kGroup = 4;
    const std::array<float, kGroup> steps = {0.f, increment, 2.f * increment, 3.f * increment};
    const float group_step = 4.f * increment;

    size_t i = 0;
    // Four independent phases break the per-sample phase-add dependency and map directly to NEON lanes.
    for (; i + kGroup <= count; i += kGroup)
    {
#ifdef SFFDN_SIMD_NEON
        const float32x4_t phases = vaddq_f32(vld1q_f32(steps.data()), vdupq_n_f32(phase));
        const float32x4_t sine = Sine4(vaddq_f32(phases, vdupq_n_f32(phase_offset)));
        vector_sink(i, simd::MulAdd(sine, simd::Splat(amplitude), simd::Splat(offset)));
#else
        for (size_t lane = 0; lane < kGroup; ++lane)
        {
            scalar_sink(i + lane, (Sine(phase + steps[lane] + phase_offset) * amplitude) + offset);
        }
#endif
        phase += group_step;
    }

    for (; i < count; ++i)
    {
        scalar_sink(i, (Sine(phase + phase_offset) * amplitude) + offset);
        phase += increment;
    }

    return phase;
}
} // namespace

SineWave::SineWave(float frequency, float initial_phase)
    : phase_(initial_phase)
    , phase_increment_(frequency)
    , amplitude_(1.0f)
    , offset_(0.0f)
    , phase_offset_(0.0f)
{
}

void SineWave::ResetPhase()
{
    phase_ = 0.0f;
}

void SineWave::SetFrequency(float frequency)
{
    phase_increment_ = frequency;
}

void SineWave::SetAmplitude(float amplitude)
{
    amplitude_ = amplitude;
}

void SineWave::SetOffset(float offset)
{
    offset_ = offset;
}

float SineWave::GetAmplitude() const
{
    return amplitude_;
}

float SineWave::GetOffset() const
{
    return offset_;
}

void SineWave::SetPhaseOffset(float phase_offset)
{
    phase_offset_ = phase_offset;
}

float SineWave::NextOut() const noexcept SFFDN_NONBLOCKING
{
    return (Sine(phase_ + phase_offset_) * amplitude_) + offset_;
}

float SineWave::Tick() noexcept SFFDN_NONBLOCKING
{
    const float out = (Sine(phase_ + phase_offset_) * amplitude_) + offset_;
    phase_ += phase_increment_;
    phase_ -= std::floor(phase_);
    return out;
}

void SineWave::Generate(std::span<float> output) noexcept SFFDN_NONBLOCKING
{
    // For small block sizes, the overhead of calling vDSP is too much. Disabled for now.
#ifdef SFFDN_USE_VDSP_DISABLED
    int32_t size = output.size();
    vDSP_vramp(&phase_, &phase_increment_, output.data(), 1, size);
    ArrayMath::Scale(output, 2.f, output);
    phase_ += phase_increment_ * size;
    phase_ -= std::floor(phase_);

    vvsinpif(output.data(), output.data(), &size);

    vDSP_vsmsa(output.data(), 1, &amplitude_, &offset_, output.data(), 1, size);
#else
    const float phase_increment = phase_increment_;
    const float phase_offset = phase_offset_;
    const float amplitude = amplitude_;
    const float offset = offset_;

    float phase = phase_;
    for (float& i : output)
    {
        i = (Sine(phase + phase_offset) * amplitude) + offset;
        phase += phase_increment;
    }

    phase_ = phase;
    phase_ -= std::floor(phase_);
#endif
}

void SineWave::Multiply(std::span<const float> input, std::span<float> output) noexcept SFFDN_NONBLOCKING
{
    assert(input.size() == output.size());

    phase_ = RunOscillator(
        input.size(), phase_, phase_increment_, {phase_offset_, amplitude_, offset_},
        [&](size_t i, simd::Vec values) {
            simd::Store(simd::LanesAt(output, i), simd::Mul(simd::Load(simd::LanesAt(input, i)), values));
        },
        [&](size_t i, float value) { output[i] = input[i] * value; });
    phase_ -= std::floor(phase_);
}

void SineWave::MultiplyAccumulate(std::span<const float> input, std::span<float> output) noexcept SFFDN_NONBLOCKING
{
    assert(input.size() == output.size());

    phase_ = RunOscillator(
        input.size(), phase_, phase_increment_, {phase_offset_, amplitude_, offset_},
        [&](size_t i, simd::Vec values) {
            simd::Store(simd::LanesAt(output, i), simd::MulAdd(simd::Load(simd::LanesAt(input, i)), values,
                                                               simd::Load(simd::LanesAt(output, i))));
        },
        [&](size_t i, float value) { output[i] += input[i] * value; });
    phase_ -= std::floor(phase_);
}

} // namespace sfFDN