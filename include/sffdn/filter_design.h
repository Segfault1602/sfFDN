// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <span>
#include <vector>

namespace sfFDN
{
void GetOnePoleAbsorption(float t60_dc, float t60_ny, float sr, float delay, float& b, float& a);

std::vector<float> GetTwoFilter(std::span<const float> t60s, float delay, float sr, float shelf_cutoff = 8000.0f);

std::vector<float> aceq(std::span<const float> mag, std::span<const float> freqs, float sr);
} // namespace sfFDN