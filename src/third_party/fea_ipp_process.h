#pragma once

#include <ipp.h>

#include "sffdn/attributes.h"

extern "C"
{
    IPPAPI(IppStatus, ippsAdd_32f_I, (const Ipp32f* pSrc, Ipp32f* pSrcDst, int len)SFFDN_NONBLOCKING)
    IPPAPI(IppStatus, ippsAdd_32f, (const Ipp32f* pSrc1, const Ipp32f* pSrc2, Ipp32f* pDst, int len)SFFDN_NONBLOCKING)
    IPPAPI(IppStatus, ippsMulC_32f, (const Ipp32f* pSrc, Ipp32f val, Ipp32f* pDst, int len)SFFDN_NONBLOCKING)
    IPPAPI(IppStatus, ippsAddProductC_32f,
           (const Ipp32f* pSrc, const Ipp32f val, Ipp32f* pSrcDst, int len)SFFDN_NONBLOCKING)
    IPPAPI(IppStatus, ippsMul_32f, (const Ipp32f* pSrc1, const Ipp32f* pSrc2, Ipp32f* pDst, int len)SFFDN_NONBLOCKING)
    IPPAPI(IppStatus, ippsFIRSR_32f,
           (const Ipp32f* pSrc, Ipp32f* pDst, int numIters, IppsFIRSpec_32f* pSpec, const Ipp32f* pDlySrc,
            Ipp32f* pDlyDst, Ipp8u* pBuf)SFFDN_NONBLOCKING)
    IPPAPI(IppStatus, ippsFIRSparse_32f,
           (const Ipp32f* pSrc, Ipp32f* pDst, int len, IppsFIRSparseState_32f* pState)SFFDN_NONBLOCKING)
}
