#pragma once

#include <cmath>

namespace fdn
{
inline void get_filter_coefficients(float t60_dc, float t60_ny, float sr, float delay, float& b, float& a)
{
    const float alpha = t60_ny / t60_dc;
    float gain = std::pow(10.f, -3.f * delay / (t60_dc * sr));
    float pole = (std::log(10.f) / 4.f) * std::log10(gain) * (1.f - 1.f / std::pow(alpha, 2.f));

    b = (1.f - pole) * gain;
    a = -pole;
}
} // namespace fdn