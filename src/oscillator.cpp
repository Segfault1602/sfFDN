#include "sffdn/oscillator.h"

#include "array_math.h"
#include "simd.h"
#include "sine_table.h"

#include <algorithm>
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
    phase = phase - std::floor(phase);

    const float index = phase * sfFDN::kSineTableSize;
    const auto raw_index = static_cast<int32_t>(index);
    const auto uindex = std::min(raw_index, static_cast<int32_t>(sfFDN::kSineTableSize - 1));
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
#ifdef SFFDN_HAS_SIMD
simd::Vec SineVector(simd::Vec phase) noexcept SFFDN_NONBLOCKING
{
    const simd::Vec wrapped = simd::Sub(phase, simd::Floor(phase));
    const simd::Vec index = simd::Mul(wrapped, simd::Splat(static_cast<float>(kSineTableSize)));
    const simd::IntVec uindex = simd::Min(simd::ToInt(index), static_cast<int32_t>(kSineTableSize - 1));
    const simd::Vec frac = simd::Sub(index, simd::ToFloat(uindex));

    const auto [a, b] = simd::GatherAdjacent(kSineTable, uindex);
    return simd::MulAdd(simd::Sub(b, a), frac, a);
}
#endif

template <typename VectorSink, typename ScalarSink>
float RunOscillator(size_t count, float phase, float increment, const std::array<float, 3>& wave,
                    [[maybe_unused]] VectorSink vector_sink, ScalarSink scalar_sink) noexcept SFFDN_NONBLOCKING
{
    const auto [phase_offset, amplitude, offset] = wave;

    constexpr size_t kGroup = simd::kWidth;

    size_t i = 0;
    for (; i + kGroup <= count; i += kGroup)
    {
#ifdef SFFDN_HAS_SIMD
        std::array<float, kGroup> phases{};
        for (float& lane_phase : phases)
        {
            lane_phase = phase + phase_offset;
            phase += increment;
        }
        const simd::Vec sine = SineVector(simd::Load(phases.data()));
        vector_sink(i, simd::MulAdd(sine, simd::Splat(amplitude), simd::Splat(offset)));
#else
        for (size_t lane = 0; lane < kGroup; ++lane)
        {
            scalar_sink(i + lane, (Sine(phase + phase_offset) * amplitude) + offset);
            phase += increment;
        }
#endif
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

    phase_ = RunOscillator(
        output.size(), phase_, phase_increment, {phase_offset, amplitude, offset},
        [&](size_t i, simd::Vec values) { simd::Store(simd::LanesAt(output, i), values); },
        [&](size_t i, float value) { output[i] = value; });
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