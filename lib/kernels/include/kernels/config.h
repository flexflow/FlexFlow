#ifndef _FLEXFLOW_KERNELS_CONFIG_H
#define _FLEXFLOW_KERNELS_CONFIG_H

#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include <cublas_v2.h>
#include <cudnn.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hipblas.h>
#include <miopen/miopen.h>
#else
#error "Unknown device"
#endif
#ifdef FF_USE_NCCL
#include <nccl.h>
#endif

#define MAX_NUM_INPUTS 256
#define MAX_NUM_WEIGHTS 64
#define MAX_NUM_OUTPUTS 256

namespace FlexFlow {

struct FFHandler {
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnHandle_t dnn;
  cublasHandle_t blas;
#else
  miopenHandle_t dnn;
  hipblasHandle_t blas;
#endif
  void *workSpace;
  size_t workSpaceSize;
  bool allowTensorOpMathConversion;
#ifdef FF_USE_NCCL
  ncclComm_t ncclComm;
#endif
};

}

#endif
