// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/time_varying_feedback_matrix.h"

#include <span>

namespace sfFDN::detail
{

class TimeVaryingFeedbackMatrixTestAccess
{
  public:
    // custom_base_matrix uses Eigen's column-major storage and bypasses the random-basis determinant correction.
    static TimeVaryingFeedbackMatrix Create(const TimeVaryingFeedbackMatrixOptions& options,
                                            std::span<const float> custom_base_matrix);
};

} // namespace sfFDN::detail
