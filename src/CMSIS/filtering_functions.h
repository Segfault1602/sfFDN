/******************************************************************************
 * @file     filtering_functions.h
 * @brief    Public header file for CMSIS DSP Library
 * @version  V1.10.0
 * @date     08 July 2021
 * Target Processor: Cortex-M and Cortex-A cores
 ******************************************************************************/
/*
 * Copyright (c) 2010-2020 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FILTERING_FUNCTIONS_H_
#define FILTERING_FUNCTIONS_H_

// #include "arm_math_memory.h"
// #include "arm_math_types.h"

// #include "dsp/none.h"
// #include "dsp/utils.h"

// #include "dsp/fast_math_functions.h"
// #include "dsp/support_functions.h"

#include <cstdint>

#include <arm_neon.h>

#define ARM_DSP_ATTRIBUTE
#define ARM_MATH_NEON

#ifdef __cplusplus
extern "C"
{
#endif

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    /**
     * @brief Instance structure for the modified Biquad coefs required by vectorized code.
     */
    typedef struct
    {
        float32_t coeffs[8][4]; /**< Points to the array of modified coefficients.  The array is of length 32. There is
                                   one per stage */
    } arm_biquad_mod_coef_f32;
#endif

    /**
     * @brief Instance structure for the floating-point transposed direct form II Biquad cascade filter.
     */
    typedef struct
    {
        uint8_t numStages; /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
        float32_t* pState; /**< points to the array of state coefficients.  The array is of length 2*numStages. */
        const float32_t* pCoeffs; /**< points to the array of coefficients.  The array is of length 5*numStages. */
    } arm_biquad_cascade_df2T_instance_f32;

    /**
     * @brief Processing function for the floating-point transposed direct form II Biquad cascade filter.
     * @param[in]  S          points to an instance of the filter data structure.
     * @param[in]  pSrc       points to the block of input data.
     * @param[out] pDst       points to the block of output data
     * @param[in]  blockSize  number of samples to process.
     */
    void arm_biquad_cascade_df2T_f32(const arm_biquad_cascade_df2T_instance_f32* S, const float32_t* pSrc,
                                     float32_t* pDst, uint32_t blockSize);

#if defined(ARM_MATH_NEON) || defined(DOXYGEN)
    /**
      @brief         Compute new coefficient arrays for use in vectorized filter (Neon only).
      @param[in]     numStages         number of 2nd order stages in the filter.
      @param[in]     pCoeffs           points to the original filter coefficients.
      @param[in]     pComputedCoeffs   points to the new computed coefficients for the vectorized version.
    */
    void arm_biquad_cascade_df2T_compute_coefs_f32(uint8_t numStages, const float32_t* pCoeffs,
                                                   float32_t* pComputedCoeffs);
#endif
    /**
     * @brief  Initialization function for the floating-point transposed direct form II Biquad cascade filter.
     * @param[in,out] S          points to an instance of the filter data structure.
     * @param[in]     numStages  number of 2nd order stages in the filter.
     * @param[in]     pCoeffs    points to the filter coefficients.
     * @param[in]     pState     points to the state buffer.
     */
    void arm_biquad_cascade_df2T_init_f32(arm_biquad_cascade_df2T_instance_f32* S, uint8_t numStages,
                                          const float32_t* pCoeffs, float32_t* pState);

#ifdef __cplusplus
}
#endif

#endif /* ifndef _FILTERING_FUNCTIONS_H_ */