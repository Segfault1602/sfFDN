#pragma once

#include <Accelerate/Accelerate.h>

#include "sffdn/attributes.h"

extern "C"
{
    extern void vDSP_vadd(const float* __A, vDSP_Stride __IA, const float* __B, vDSP_Stride __IB, float* __C,
                          vDSP_Stride __IC, vDSP_Length __N) SFFDN_NONBLOCKING;
    extern void vDSP_vsmul(const float* __A, vDSP_Stride __IA, const float* __B, float* __C, vDSP_Stride __IC,
                           vDSP_Length __N) SFFDN_NONBLOCKING;
    extern void vDSP_vsma(const float* __A, vDSP_Stride __IA, const float* __B, const float* __C, vDSP_Stride __IC,
                          float* __D, vDSP_Stride __ID, vDSP_Length __N) SFFDN_NONBLOCKING;
    extern void vDSP_biquadm(vDSP_biquadm_Setup __Setup, const float* __nonnull* __nonnull __X, vDSP_Stride __IX,
                             float* __nonnull* __nonnull __Y, vDSP_Stride __IY, vDSP_Length __N) SFFDN_NONBLOCKING;
    extern void vDSP_biquad(const struct vDSP_biquad_SetupStruct* __nonnull __Setup, float* __nonnull __Delay,
                            const float* __nonnull __X, vDSP_Stride __IX, float* __nonnull __Y, vDSP_Stride __IY,
                            vDSP_Length __N) SFFDN_NONBLOCKING;
}
