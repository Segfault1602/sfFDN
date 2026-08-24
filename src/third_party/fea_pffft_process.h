#pragma once

#include <pffft.h>

#include "sffdn/attributes.h"

extern "C"
{
    void pffft_transform(PFFFT_Setup* setup, const float* input, float* output, float* work,
                         pffft_direction_t direction) SFFDN_NONBLOCKING;
    void pffft_zconvolve_accumulate(PFFFT_Setup* setup, const float* dft_a, const float* dft_b, float* dft_ab,
                                    float scaling) SFFDN_NONBLOCKING;
}
