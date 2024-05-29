#ifndef _FLEXFLOW_KERNELS_FF_HANDLE_H
#define _FLEXFLOW_KERNELS_FF_HANDLE_H

#ifdef FF_USE_HIP_ROCM
#include <rccl/rccl.h>
#elif FF_USE_NCCL
#include <nccl.h>
#endif

#include "device.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct PerDeviceFFHandle {
  ffHandle_t dnn;
  ffblasHandle_t blas;

  void *workSpace;
  size_t workSpaceSize;
  bool allowTensorOpMathConversion;

#if defined(FF_USE_HIP_ROCM) || defined(FF_USE_NCCL)
  ncclComm_t ncclComm;
#endif
};

#if defined(FF_USE_HIP_ROCM) || defined(FF_USE_NCCL)
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(PerDeviceFFHandle,
                                             dnn,
                                             blas,
                                             workSpace,
                                             workSpaceSize,
                                             allowTensorOpMathConversion,
                                             ncclComm);
#else
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(PerDeviceFFHandle,
                                             dnn,
                                             blas,
                                             workSpace,
                                             workSpaceSize,
                                             allowTensorOpMathConversion);
#endif

} // namespace FlexFlow

#endif
