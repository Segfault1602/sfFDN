#include "filter.h"

#include <cmath>
#include <iostream>
#include <numbers>

namespace
{
constexpr float TWO_PI = std::numbers::pi_v<float> * 2;
}

namespace fdn
{
void Filter::SetGain(float gain)
{
    gain_ = gain;
}

void Filter::SetA(const float (&a)[COEFFICIENT_COUNT])
{
    for (size_t i = 0; i < COEFFICIENT_COUNT; ++i)
    {
        a_[i] = a[i];
    }
}

void Filter::SetB(const float (&b)[COEFFICIENT_COUNT])
{
    for (size_t i = 0; i < COEFFICIENT_COUNT; ++i)
    {
        b_[i] = b[i];
    }
}

void Filter::Clear()
{
    for (size_t i = 0; i < COEFFICIENT_COUNT; ++i)
    {
        outputs_[i] = 0.f;
        inputs_[i] = 0.f;
    }
}

void Filter::ProcessBlock(const float* in, float* out, size_t size)
{
    assert(in != nullptr);
    assert(out != nullptr);

    for (size_t i = 0; i < size; ++i)
    {
        out[i] = Tick(in[i]);
    }
}

void OnePoleFilter::SetPole(float pole)
{
    // https://ccrma.stanford.edu/~jos/fp/One_Pole.html
    // If the filter has a pole at z = -a, then a_[1] will be -pole;
    assert(pole <= 1.f && pole >= -1.f);

    // Set the b value to 1 - |a| to get a peak gain of 1.
    b_[0] = 1.f - std::abs(pole);
    a_[1] = -pole;
}

void OnePoleFilter::SetCoefficients(float b0, float a1)
{
    b_[0] = b0;
    a_[1] = a1;
}

void OnePoleFilter::SetDecayFilter(float decayDb, float timeMs, float samplerate)
{
    assert(decayDb < 0.f);
    const float lambda = std::log(std::pow(10.f, (decayDb / 20.f)));
    const float pole = std::exp(lambda / (timeMs / 1000.f) / samplerate);
    SetPole(pole);
}

void OnePoleFilter::SetLowpass(float cutoff)
{
    assert(cutoff >= 0.f && cutoff <= 1.f);
    const float wc = TWO_PI * cutoff;
    const float y = 1 - std::cos(wc);
    const float p = -y + std::sqrt(y * y + 2 * y);
    SetPole(1 - p);
}

float OnePoleFilter::Tick(float in)
{
    outputs_[0] = gain_ * in * b_[0] - outputs_[1] * a_[1];
    outputs_[1] = outputs_[0];
    return outputs_[0];
}

float OneZeroFilter::Tick(float in)
{
    float out = gain_ * in * b_[0] + inputs_[0] * b_[1];
    inputs_[0] = in;
    return out;
}

float TwoPoleFilter::Tick(float in)
{
    outputs_[0] = gain_ * in * b_[0] - outputs_[1] * a_[1] - outputs_[2] * a_[2];
    outputs_[2] = outputs_[1];
    outputs_[1] = outputs_[0];
    return outputs_[0];
}

float TwoZeroFilter::Tick(float in)
{
    float out = gain_ * in * b_[0] + inputs_[0] * b_[1] + inputs_[1] * b_[2];
    inputs_[1] = inputs_[0];
    inputs_[0] = in;
    return out;
}

void Biquad::SetCoefficients(float b0, float b1, float b2, float a1, float a2)
{
    b_[0] = b0;
    b_[1] = b1;
    b_[2] = b2;
    a_[1] = a1;
    a_[2] = a2;
}

float Biquad::Tick(float in)
{
    inputs_[0] = gain_ * in;
    outputs_[0] = inputs_[0] * b_[0] + inputs_[1] * b_[1] + inputs_[2] * b_[2];
    outputs_[0] -= outputs_[1] * a_[1] + outputs_[2] * a_[2];

    inputs_[2] = inputs_[1];
    inputs_[1] = inputs_[0];

    outputs_[2] = outputs_[1];
    outputs_[1] = outputs_[0];
    return outputs_[0];
}

void Biquad2::SetCoefficients(float b0, float b1, float b2, float a1, float a2)
{
    b_[0] = b0;
    b_[1] = b1;
    b_[2] = b2;
    a_[1] = a1;
    a_[2] = a2;
}

float Biquad2::Tick(float in)
{
    outputs_[0] = b_[0] * in + d1_;
    d1_ = b_[1] * in - a_[1] * outputs_[0] + d2_;
    d2_ = b_[2] * in - a_[2] * outputs_[0];
    return outputs_[0];
}

// void Biquad::ProcessBlock(const float* in, float* out, size_t size)
// {
//     assert(in != nullptr);
//     assert(out != nullptr);

//     for (size_t i = 0; i < size; ++i)
//     {
//         inputs_[0] = gain_ * in[i];
//         outputs_[0] = inputs_[0] * b_[0] + inputs_[1] * b_[1] + inputs_[2] * b_[2];
//         outputs_[0] -= outputs_[1] * a_[1] + outputs_[2] * a_[2];

//         inputs_[2] = inputs_[1];
//         inputs_[1] = inputs_[0];

//         outputs_[2] = outputs_[1];
//         outputs_[1] = outputs_[0];
//         out[i] = outputs_[0];
//     }
// }

CascadedBiquads::~CascadedBiquads()
{
    vDSP_biquad_DestroySetup(biquad_setup_);
    biquad_setup_ = nullptr;
}

void CascadedBiquads::Clear()
{
    state_.clear();
    state_.resize(stage_ * 2, 0);

    delays_.clear();
    delays_.resize(stage_ * 2 + 2, 0);
}

void CascadedBiquads::SetCoefficients(size_t num_stage, std::span<const float> coeffs)
{
    assert(coeffs.size() == num_stage * 5);

    coeffs_.clear();

    coeffs_.insert(coeffs_.begin(), coeffs.begin(), coeffs.end());
    stage_ = num_stage;

    state_.resize(num_stage * 2, 0);

    if (biquad_setup_ != nullptr)
    {
        vDSP_biquad_DestroySetup(biquad_setup_);
        biquad_setup_ = nullptr;
    }

    std::vector<double> coeffs_d(num_stage * 5);
    for (size_t i = 0; i < coeffs.size(); i++)
    {
        coeffs_d[i] = coeffs_[i];
    }

    biquad_setup_ = vDSP_biquad_CreateSetup(coeffs_d.data(), num_stage);
    assert(biquad_setup_ != nullptr);

    delays_.clear();
    delays_.resize(num_stage * 2 + 2, 0);
}

float CascadedBiquads::Tick(float in)
{
    float out = 0;
    ProcessBlock(&in, &out, 1);
    return out;
}

#if 0
void CascadedBiquads::ProcessBlock(const float* in, float* out, size_t size)
{
    assert(in != nullptr);
    assert(out != nullptr);

    size_t stage = stage_;

    const float* in_ptr = in;
    float* out_ptr = out;

    size_t sample = size;
    while (sample > 0)
    {
        float* coeffs_ptr = coeffs_.data();
        float* state_ptr = state_.data();
        stage = stage_;
        float in1 = *in_ptr++;
        float out1 = 0;
        do
        {
            float* b = coeffs_ptr;
            coeffs_ptr += 3;
            float* a = coeffs_ptr;
            coeffs_ptr += 2;

            float* state = state_ptr;
            state_ptr += 2;

            out1 = b[0] * in1 + state[0];
            state[0] = b[1] * in1 - a[0] * out1 + state[1];
            state[1] = b[2] * in1 - a[1] * out1;

            in1 = out1;
            --stage;
        } while (stage > 0);

        *out_ptr++ = out1;

        --sample;
    }
}

void CascadedBiquads::ProcessBlock(const float* in, float* out, size_t size)
{
    assert(in != nullptr);
    assert(out != nullptr);

    size_t stage = stage_;

    const float* in_ptr = in;
    float* out_ptr = out;
    float* coeffs_ptr = coeffs_.data();
    float* state_ptr = state_.data();

    do
    {
        float* b = coeffs_ptr;
        coeffs_ptr += 3;
        float* a = coeffs_ptr;
        coeffs_ptr += 2;

        float d1 = state_ptr[0];
        float d2 = state_ptr[1];

        size_t sample = size;
        while (sample > 0)
        {
            float in1 = *in_ptr++;

            float acc1 = b[0] * in1 + d1;

            d1 = b[1] * in1 + d2;
            d1 -= a[0] * acc1;

            d2 = b[2] * in1;
            d2 -= a[1] * acc1;

            *out_ptr++ = acc1;
            --sample;
        }

        state_ptr[0] = d1;
        state_ptr[1] = d2;
        state_ptr += 2;

        in_ptr = out;
        out_ptr = out;

        --stage;
    } while (stage > 0);
}
#endif

void CascadedBiquads::ProcessBlock(const float* in, float* out, size_t size)
{
    assert(in != nullptr);
    assert(out != nullptr);
    assert(biquad_setup_ != nullptr);

    vDSP_biquad(biquad_setup_, delays_.data(), in, 1, out, 1, size);
}

void CascadedBiquads::dump_coeffs()
{
    for (size_t i = 0; i < stage_; i++)
    {
        size_t offset = i * 5;
        std::cout << "[" << coeffs_[offset] << ", " << coeffs_[offset + 1] << ", " << coeffs_[offset + 2] << ", "
                  << coeffs_[offset + 3] << ", " << coeffs_[offset + 4] << "]" << std::endl;
    }
}

} // namespace fdn