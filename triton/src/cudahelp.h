/* Copyright 2022 NVIDIA CORPORATION
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LEGION_TRITON_CUDAHELP_H__
#define __LEGION_TRITON_CUDAHELP_H__

#define __CUDA_NO_HALF_OPERATORS__

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cudnn.h"
#include "types.h"

#define FatalError(s)                                                 \
  do {                                                                \
    std::stringstream _message;                                       \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
    std::cerr << _message.str() << "\nAborting...\n";                 \
    abort();                                                          \
  } while (0)

#define CHECK_CUDNN(cmd)                              \
  do {                                                \
    cudnnStatus_t status = (cmd);                     \
    if (status != CUDNN_STATUS_SUCCESS) {             \
      std::stringstream _error;                       \
      _error << "CUDNN failure (" << status           \
             << "): " << cudnnGetErrorString(status); \
      FatalError(_error.str());                       \
    }                                                 \
  } while (0)

#define CHECK_CURAND(cmd)                     \
  do {                                        \
    curandStatus_t status = (cmd);            \
    if (status != CURAND_STATUS_SUCCESS) {    \
      std::stringstream _error;               \
      _error << "CURAND failure: " << status; \
      FatalError(_error.str());               \
    }                                         \
  } while (0)

#define CHECK_CUDA(cmd)                              \
  do {                                               \
    cudaError_t status = (cmd);                      \
    if (status != cudaSuccess) {                     \
      std::stringstream _error;                      \
      _error << "CUDA failure (" << status           \
             << "): " << cudaGetErrorString(status); \
      FatalError(_error.str());                      \
    }                                                \
  } while (0)

#define CHECK_CUBLAS(cmd)                     \
  do {                                        \
    cublasStatus_t status = (cmd);            \
    if (status != CUBLAS_STATUS_SUCCESS) {    \
      std::stringstream _error;               \
      _error << "CUBLAS failure: " << status; \
      FatalError(_error.str());               \
    }                                         \
  } while (0)

#define CHECK_NCCL(cmd)                              \
  do {                                               \
    ncclResult_t status = (cmd);                     \
    if (status != ncclSuccess) {                     \
      std::stringstream _error;                      \
      _error << "NCCL failure (" << status           \
             << "): " << ncclGetErrorString(status); \
      FatalError(_error.str());                      \
    }                                                \
  } while (0)

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

namespace triton { namespace backend { namespace legion {

inline cudnnDataType_t
to_cudnn_datatype(DataType type)
{
  switch (type) {
    case DT_HALF:
      return CUDNN_DATA_HALF;
    case DT_FLOAT:
      return CUDNN_DATA_FLOAT;
    case DT_DOUBLE:
      return CUDNN_DATA_DOUBLE;
    case DT_INT8:
      return CUDNN_DATA_INT8;
    case DT_INT32:
      return CUDNN_DATA_INT32;
    case DT_UINT8:
      return CUDNN_DATA_UINT8;
    default:
      abort();
  }
  return CUDNN_DATA_FLOAT;
}

inline cudaDataType_t
to_cuda_datatype(DataType type)
{
  switch (type) {
    case DT_HALF:
      return CUDA_R_16F;
    case DT_FLOAT:
      return CUDA_R_32F;
    case DT_DOUBLE:
      return CUDA_R_64F;
    case DT_INT8:
      return CUDA_R_8I;
    case DT_INT32:
      return CUDA_R_32I;
    case DT_UINT8:
      return CUDA_R_8U;
    case DT_UINT32:
      return CUDA_R_32U;
    default:
      abort();
  }
  return CUDA_R_32F;
}

// To use cudnnOpTensor(), some data type combination is required,
// CUDNN_DATA_UINT8 will be returned if the combination is not supported
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnOpTensor
inline cudnnDataType_t
to_op_tensor_comp_type(DataType input0, DataType input1, DataType output)
{
  if (input0 != input1) {
    return CUDNN_DATA_UINT8;
  }
  switch (output) {
    case DataType::DT_FLOAT: {
      switch (input0) {
        case DataType::DT_FLOAT:
        case DataType::DT_HALF:
        case DataType::DT_INT8:
          return CUDNN_DATA_FLOAT;
        default:
          return CUDNN_DATA_UINT8;
      }
    }
    case DataType::DT_HALF:
    case DataType::DT_INT8:
      return ((input0 == output) || (input0 == DataType::DT_FLOAT))
                 ? CUDNN_DATA_FLOAT
                 : CUDNN_DATA_UINT8;
    case DataType::DT_DOUBLE:
      return (input0 == DataType::DT_DOUBLE) ? CUDNN_DATA_DOUBLE
                                             : CUDNN_DATA_UINT8;
    case DataType::DT_INT16:
    case DataType::DT_INT32:
    case DataType::DT_INT64:
    case DataType::DT_UINT8:
    case DataType::DT_UINT16:
    case DataType::DT_UINT32:
    case DataType::DT_UINT64:
    case DataType::DT_BOOLEAN:
    case DataType::DT_NONE:
      return CUDNN_DATA_UINT8;
  }
  return CUDNN_DATA_UINT8;
}

inline cudnnStatus_t
cudnnSetTensorDescriptorFromDomain(
    cudnnTensorDescriptor_t tensor, Legion::Domain domain, DataType type,
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW)
{
  int dims[4];
  switch (domain.get_dim()) {
    case 1: {
      Legion::Rect<1> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      return cudnnSetTensor4dDescriptor(
          tensor, format, to_cudnn_datatype(type), 1, 1, 1, dims[0]);
    }
    case 2: {
      Legion::Rect<2> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      return cudnnSetTensor4dDescriptor(
          tensor, format, to_cudnn_datatype(type), 1, 1, dims[0], dims[1]);
    }
    case 3: {
      Legion::Rect<3> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      dims[2] = rect.hi[2] - rect.lo[2] + 1;
      return cudnnSetTensor4dDescriptor(
          tensor, format, to_cudnn_datatype(type), 1, dims[0], dims[1],
          dims[2]);
    }
    case 4: {
      Legion::Rect<4> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      dims[2] = rect.hi[2] - rect.lo[2] + 1;
      dims[3] = rect.hi[3] - rect.lo[3] + 1;
      return cudnnSetTensor4dDescriptor(
          tensor, format, to_cudnn_datatype(type), dims[0], dims[1], dims[2],
          dims[3]);
    }
    default:
      abort();
  }
  return CUDNN_STATUS_BAD_PARAM;
}

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_CUDAHELP_H__
