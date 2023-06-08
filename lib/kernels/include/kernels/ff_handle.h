#ifndef _FLEXFLOW_KERNELS_FF_HANDLE_H
#define _FLEXFLOW_KERNELS_FF_HANDLE_H

#ifdef FF_USE_NCCL
#include <nccl.h>
#endif

#include "kernels/device.h"

namespace FlexFlow {

struct PerDeviceFFHandle {
  ffHandle_t dnn;
  ffblasHandle_t blas;

  void *workSpace;
  size_t workSpaceSize;
  bool allowTensorOpMathConversion;

#ifdef FF_USE_NCCL
  ncclComm_t ncclComm;
#endif
};

} // namespace FlexFlow

#endif
