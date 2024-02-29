#include "flexflow/model.h"
#include "flexflow/utils/cuda_helper.h"
#ifdef FF_USE_CUDA
#include "realm/cuda/cuda_module.h"
#elif FF_USE_HIP_CUDA
#include "realm/hip/hip_module.h"
#else
#error "Unknown device, please make sure if CUDA is enabled"
#endif

using Legion::coord_t;
using Legion::Domain;
using Legion::Rect;

namespace FlexFlow {

#ifdef FF_USE_CUDA
cudaError_t get_legion_stream(cudaStream_t *stream) {
  *stream = Realm::Cuda::get_task_cuda_stream();
  Realm::Cuda::set_task_ctxsync_required(false);
  assert(*stream != 0);
  return cudaSuccess;
}
#elif FF_USE_HIP_CUDA
cudaError_t get_legion_stream(cudaStream_t *stream) {
  *stream = Realm::Hip::get_task_hip_stream();
  Realm::Hip::set_task_ctxsync_required(false);
  assert(*stream != 0);
  return cudaSuccess;
}
#else
#error "Unknown device, please make sure if CUDA is enabled"
#endif

}; // namespace FlexFlow

using FlexFlow::get_legion_stream;

__global__ void scale_kernel(float *ptr, coord_t size, float a, float b) {
  CUDA_KERNEL_LOOP(i, size) {
    ptr[i] = (b - a) * ptr[i] + a;
  }
}

__global__ void ones_kernel(float *ptr, coord_t size) {
  CUDA_KERNEL_LOOP(i, size) {
    ptr[i] = 1.0f;
  }
}

template <typename DT>
__global__ void assign_kernel(DT *ptr, coord_t size, DT value) {
  CUDA_KERNEL_LOOP(i, size) {
    ptr[i] = value;
  }
}

template <typename DT>
__global__ void copy_kernel(DT *dst, const DT *src, coord_t size) {
  CUDA_KERNEL_LOOP(i, size) {
    dst[i] = src[i];
  }
}
template <typename DT>
__global__ void copy_kernel_with_replicate(DT *dst,
                                           const DT *src,
                                           coord_t origin_size,
                                           coord_t size) {
  CUDA_KERNEL_LOOP(i, size) {
    dst[i] = src[i % origin_size];
  }
}

template <typename DT>
__global__ void reluBackward(DT *grad_ptr, const DT *output, size_t n) {
  CUDA_KERNEL_LOOP(i, n) {
    grad_ptr[i] = (output[i] > 0.0f) ? grad_ptr[i] : 0;
  }
}

__host__ void relu_backward_kernel(DataType data_type,
                                   void *output_grad_ptr,
                                   void const *output_ptr,
                                   size_t output_size,
                                   cudaStream_t stream) {
  if (data_type == DT_FLOAT) {
    reluBackward<float>
        <<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
            (float *)output_grad_ptr, (float const *)output_ptr, output_size);
  } else if (data_type == DT_DOUBLE) {
    reluBackward<double>
        <<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
            (double *)output_grad_ptr, (double const *)output_ptr, output_size);
  } else {
    assert(false && "Unsupported data type in Linear backward");
    exit(1);
  }
}

template <typename DT>
__global__ void
    sigmoid_backward_function(DT *grad_ptr, const DT *output, size_t n) {
  CUDA_KERNEL_LOOP(i, n) {
    grad_ptr[i] = grad_ptr[i] * output[i] * (1.0f - output[i]);
  }
}

__host__ void sigmoid_backward_kernel(DataType data_type,
                                      void *output_grad_ptr,
                                      void const *output_ptr,
                                      size_t output_size,
                                      cudaStream_t stream) {
  if (data_type == DT_FLOAT) {
    sigmoid_backward_function<float>
        <<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
            (float *)output_grad_ptr, (float const *)output_ptr, output_size);
  } else if (data_type == DT_DOUBLE) {
    sigmoid_backward_function<double>
        <<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
            (double *)output_grad_ptr, (double const *)output_ptr, output_size);
  } else {
    assert(false && "Unsupported data type in Linear backward");
    exit(1);
  }
}

__global__ void gelu_forward_kernel(size_t size,
                                    float const B,
                                    float const C,
                                    float *input) {
  CUDA_KERNEL_LOOP(i, size) {
    float const in = input[i];
    float const cdf = 0.5f + 0.5f * tanh(in * (C * in * in + B));
    input[i] = in * cdf;
  }
}

__global__ void
    apply_add(float *data_ptr, float const *replica_ptr, size_t size) {
  CUDA_KERNEL_LOOP(i, size) {
    data_ptr[i] += replica_ptr[i];
  }
}

template <typename T>
__global__ void
    apply_add_with_scale(T *data_ptr, T const *grad_ptr, size_t size, T scale) {
  CUDA_KERNEL_LOOP(i, size) {
    data_ptr[i] += grad_ptr[i] * scale;
  }
}

template <typename T>
__global__ void add_kernel(T *data_ptr, T const *grad_ptr, size_t size) {
  CUDA_KERNEL_LOOP(i, size) {
    data_ptr[i] += grad_ptr[i];
  }
}

__global__ void add_with_stride(float *output,
                                float const *input,
                                int num_blocks,
                                int output_blk_size,
                                int input_blk_size) {
  int min_blk_size = min(output_blk_size, input_blk_size);
  CUDA_KERNEL_LOOP(i, num_blocks * min_blk_size) {
    int blk_idx = i / min_blk_size;
    int blk_offset = i % min_blk_size;
    int input_offset = blk_idx * input_blk_size + blk_offset;
    int output_offset = blk_idx * output_blk_size + blk_offset;
    output[output_offset] += input[input_offset];
  }
}

__global__ void copy_with_stride(float *output,
                                 float const *input,
                                 int num_blocks,
                                 int output_blk_size,
                                 int input_blk_size) {
  int min_blk_size = min(output_blk_size, input_blk_size);
  CUDA_KERNEL_LOOP(i, num_blocks * min_blk_size) {
    int blk_idx = i / min_blk_size;
    int blk_offset = i % min_blk_size;
    int input_offset = blk_idx * input_blk_size + blk_offset;
    int output_offset = blk_idx * output_blk_size + blk_offset;
    output[output_offset] = input[input_offset];
  }
}

__host__ void updateGAS(float *para_ptr,
                        float const *grad_ptr,
                        size_t replica_size,
                        int num_replica,
                        float learning_rate) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // Step 1: gater gradients to the first replica
  for (int i = 1; i < num_replica; i++) {
    float const *replica = grad_ptr + i * replica_size;
    apply_add<<<GET_BLOCKS(replica_size), CUDA_NUM_THREADS, 0, stream>>>(
        (float *)grad_ptr, replica, replica_size);
  }
  // Step 2: scale the first replica
  float scale_factor = 1.0f / num_replica * (-learning_rate);
  apply_add_with_scale<<<GET_BLOCKS(replica_size),
                         CUDA_NUM_THREADS,
                         0,
                         stream>>>(
      para_ptr, grad_ptr, replica_size, scale_factor);
}

template <typename T>
__host__ void
    print_tensor(T const *ptr, size_t num_elements, char const *prefix) {
  // device synchronize to make sure the data are ready
  // checkCUDA(cudaDeviceSynchronize());
  T *host_ptr;
  checkCUDA(cudaHostAlloc(&host_ptr,
                          sizeof(T) * num_elements,
                          cudaHostAllocPortable | cudaHostAllocMapped));
  checkCUDA(cudaMemcpy(
      host_ptr, ptr, sizeof(T) * num_elements, cudaMemcpyDeviceToHost));
  // checkCUDA(cudaDeviceSynchronize());
  int idx = 0;
  printf("%s", prefix);
  for (idx = 0; idx < num_elements; idx++) {
    printf(" %.10lf", (float)host_ptr[idx]);
    if (idx >= 16) {
      break;
    }
  }
  printf("\n");
  checkCUDA(cudaFreeHost(host_ptr));
}

template <typename T>
__host__ void
    save_tensor(T const *ptr, size_t num_elements, char const *file_name) {
  T *host_ptr;
  checkCUDA(cudaHostAlloc(&host_ptr,
                          sizeof(T) * num_elements,
                          cudaHostAllocPortable | cudaHostAllocMapped));
  checkCUDA(cudaMemcpy(
      host_ptr, ptr, sizeof(T) * num_elements, cudaMemcpyDeviceToHost));
  FILE *tensor_file;
  tensor_file = fopen(file_name, "w");
  for (unsigned i = 0; i < num_elements; i++) {
    fprintf(tensor_file, "%.8f, ", (float)host_ptr[i]);
  }

  fclose(tensor_file);
  checkCUDA(cudaFreeHost(host_ptr));
}

cudnnStatus_t cudnnSetTensorDescriptorFromDomain4SoftMax(
    cudnnTensorDescriptor_t tensor, Domain domain, DataType data_type) {
  int dims[MAX_TENSOR_DIM];
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(data_type);
  switch (domain.get_dim()) {
    case 1: {
      Rect<1> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      return cudnnSetTensor4dDescriptor(
          tensor, CUDNN_TENSOR_NCHW, cudnn_data_type, dims[0], 1, 1, 1);
    }
    case 2: {
      Rect<2> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      return cudnnSetTensor4dDescriptor(
          tensor, CUDNN_TENSOR_NCHW, cudnn_data_type, dims[1], dims[0], 1, 1);
    }
    case 3: {
      Rect<3> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      dims[2] = rect.hi[2] - rect.lo[2] + 1;
      return cudnnSetTensor4dDescriptor(tensor,
                                        CUDNN_TENSOR_NCHW,
                                        cudnn_data_type,
                                        dims[2] * dims[1],
                                        dims[0],
                                        1,
                                        1);
    }
    case 4: {
      Rect<4> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      dims[2] = rect.hi[2] - rect.lo[2] + 1;
      dims[3] = rect.hi[3] - rect.lo[3] + 1;
      return cudnnSetTensor4dDescriptor(tensor,
                                        CUDNN_TENSOR_NCHW,
                                        cudnn_data_type,
                                        dims[3] * dims[2] * dims[1],
                                        dims[0],
                                        1,
                                        1);
    }
    default:
      assert(false && "Unsupported dim number");
  }
  return CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t cudnnSetTensorDescriptorFromDomain(cudnnTensorDescriptor_t tensor,
                                                 Domain domain) {
  int dims[MAX_TENSOR_DIM];
  switch (domain.get_dim()) {
    case 1: {
      Rect<1> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      return cudnnSetTensor4dDescriptor(
          tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims[0], 1, 1, 1);
    }
    case 2: {
      Rect<2> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      return cudnnSetTensor4dDescriptor(
          tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims[1], dims[0], 1, 1);
    }
    case 3: {
      Rect<3> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      dims[2] = rect.hi[2] - rect.lo[2] + 1;
      return cudnnSetTensor4dDescriptor(tensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        dims[2],
                                        dims[1],
                                        dims[0],
                                        1);
    }
    case 4: {
      Rect<4> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      dims[2] = rect.hi[2] - rect.lo[2] + 1;
      dims[3] = rect.hi[3] - rect.lo[3] + 1;
      return cudnnSetTensor4dDescriptor(tensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        dims[3],
                                        dims[2],
                                        dims[1],
                                        dims[0]);
    }
    case 5: {
      Rect<5> rect = domain;
      int leading_dim_size = rect.hi[4] - rect.lo[4] + 1;
      assert(leading_dim_size == 1);
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      dims[2] = rect.hi[2] - rect.lo[2] + 1;
      dims[3] = rect.hi[3] - rect.lo[3] + 1;
      return cudnnSetTensor4dDescriptor(tensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        dims[3],
                                        dims[2],
                                        dims[1],
                                        dims[0]);
    }
    default:
      assert(false && "Unsupported dim number");
  }
  return CUDNN_STATUS_BAD_PARAM;
}

cudnnDataType_t ff_to_cudnn_datatype(DataType type) {
  switch (type) {
    case DT_FLOAT:
      return CUDNN_DATA_FLOAT;
    case DT_DOUBLE:
      return CUDNN_DATA_DOUBLE;
    case DT_INT32:
      return CUDNN_DATA_INT32;
    default:
      assert(false && "Unsupported cudnn data type");
  }
  return CUDNN_DATA_FLOAT;
}

cudaDataType_t ff_to_cuda_datatype(DataType type) {
  switch (type) {
    case DT_FLOAT:
      return CUDA_R_32F;
    case DT_DOUBLE:
      return CUDA_R_64F;
    case DT_INT32:
      return CUDA_R_32I;
    default:
      assert(false && "Unspoorted cuda data type");
  }
  return CUDA_R_32F;
}

template __global__ void
    assign_kernel<half>(half *ptr, coord_t size, half value);
template __global__ void
    assign_kernel<float>(float *ptr, coord_t size, float value);
template __global__ void
    assign_kernel<double>(double *ptr, coord_t size, double value);
template __global__ void
    assign_kernel<int32_t>(int32_t *ptr, coord_t size, int32_t value);
template __global__ void
    assign_kernel<int64_t>(int64_t *ptr, coord_t size, int64_t value);

template __global__ void
    add_kernel<float>(float *dst, float const *src, size_t size);
template __global__ void
    add_kernel<double>(double *dst, double const *src, size_t size);
template __global__ void
    add_kernel<int32_t>(int32_t *dst, int32_t const *src, size_t size);
template __global__ void
    add_kernel<int64_t>(int64_t *dst, int64_t const *src, size_t size);

template __global__ void
    copy_kernel<float>(float *dst, float const *src, coord_t size);
template __global__ void copy_kernel_with_replicate<float>(float *dst,
                                                           float const *src,
                                                           coord_t origin_size,
                                                           coord_t size);
template __global__ void copy_kernel_with_replicate<int32_t>(
    int32_t *dst, int32_t const *src, coord_t origin_size, coord_t size);
template __global__ void copy_kernel_with_replicate<int64_t>(
    int64_t *dst, int64_t const *src, coord_t origin_size, coord_t size);
template __global__ void
    copy_kernel<int32_t>(int32_t *dst, int32_t const *src, coord_t size);
template __global__ void
    copy_kernel<int64_t>(int64_t *dst, int64_t const *src, coord_t size);

template __global__ void apply_add_with_scale<float>(float *data_ptr,
                                                     float const *grad_ptr,
                                                     size_t size,
                                                     float scale);
template __global__ void apply_add_with_scale<double>(double *data_ptr,
                                                      double const *grad_ptr,
                                                      size_t size,
                                                      double scale);
template __global__ void apply_add_with_scale<int32_t>(int32_t *data_ptr,
                                                       int32_t const *grad_ptr,
                                                       size_t size,
                                                       int32_t scale);
template __global__ void apply_add_with_scale<int64_t>(int64_t *data_ptr,
                                                       int64_t const *grad_ptr,
                                                       size_t size,
                                                       int64_t scale);

template __host__ void
    print_tensor<float>(float const *ptr, size_t rect, char const *prefix);
template __host__ void
    print_tensor<double>(double const *ptr, size_t rect, char const *prefix);
template __host__ void
    print_tensor<int32_t>(int32_t const *ptr, size_t rect, char const *prefix);
template __host__ void
    print_tensor<int64_t>(int64_t const *ptr, size_t rect, char const *prefix);
template __host__ void
    save_tensor<float>(float const *ptr, size_t rect, char const *file_name);
template __host__ void save_tensor<int32_t>(int32_t const *ptr,
                                            size_t rect,
                                            char const *file_name);
