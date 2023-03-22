#ifndef _FLEXFLOW_KERNELS_CONFIG_H
#define _FLEXFLOW_KERNELS_CONFIG_H

#include "kernels/device.h"

#ifdef FF_USE_NCCL
#include <nccl.h>
#endif

#if defined(FF_USE_CUDA)
#include <cuda_fp16.h>
#elif defined(FF_USE_HIP_CUDA)
#include <cuda_fp16.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hip/hip_fp16.h>
#endif


#define MAX_NUM_INPUTS 256
#define MAX_NUM_WEIGHTS 64
#define MAX_NUM_OUTPUTS 256
#define MAX_OPNAME 128

namespace FlexFlow {

struct FFHandler {
  ffHandle_t dnn;
  ffblasHandle_t blas;

  void *workSpace;
  size_t workSpaceSize;
  bool allowTensorOpMathConversion;

#ifdef FF_USE_NCCL
  ncclComm_t ncclComm;
#endif
};

}

#endif
