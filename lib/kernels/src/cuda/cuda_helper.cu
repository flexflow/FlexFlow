#include "device.h"
#include "kernels/datatype_dispatch.h"

namespace FlexFlow {

#ifdef FF_USE_CUDA
cudaError_t get_legion_stream(cudaStream_t *stream) {
#ifdef DISABLE_LEGION_CUDA_HIJACK
  *stream = (cudaStream_t)0;
  return cudaSuccess;
#else
  return cudaStreamCreate(stream);
#endif
}
#elif FF_USE_HIP_CUDA
extern "C" {
cudaStream_t hipGetTaskStream();
}
cudaError_t get_legion_stream(cudaStream_t *stream) {
#ifdef DISABLE_LEGION_CUDA_HIJACK
  *stream = (cudaStream_t)0;
#else
  *stream = hipGetTaskStream();
#endif
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
__global__ void assign_kernel(DT *ptr, size_t size, DT value) {
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
  if (data_type == DataType::FLOAT) {
    reluBackward<float>
        <<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
            (float *)output_grad_ptr, (float const *)output_ptr, output_size);
  } else if (data_type == DataType::DOUBLE) {
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
  if (data_type == DataType::FLOAT) {
    sigmoid_backward_function<float>
        <<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
            (float *)output_grad_ptr, (float const *)output_ptr, output_size);
  } else if (data_type == DataType::DOUBLE) {
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
    printf(" %.4lf", (float)host_ptr[idx]);
    if (idx >= 16) {
      break;
    }
  }
  printf("\n");
  checkCUDA(cudaFreeHost(host_ptr));
}

ffStatus_t
    cudnnSetTensorDescriptorFromArrayShape(cudnnTensorDescriptor_t tensor,
                                           ArrayShape const &shape) {
  std::vector<std::size_t> reversed_dims(shape.dims.begin(), shape.dims.end());
  reversed(reversed_dims);
  ArrayShape flipped(reversed_dims);

  if (flipped.get_dim() == 5) {
    assert(flipped[legion_dim_t(0)] == 1);
    flipped = flipped.sub_shape(legion_dim_t(1), std::nullopt);
  }

  assert(flipped.get_dim() > 0);
  assert(flipped.get_dim() < 4);

  return cudnnSetTensor4dDescriptor(tensor,
                                    CUDNN_TENSOR_NCHW,
                                    CUDNN_DATA_FLOAT,
                                    flipped.at_maybe(0).value_or(1),
                                    flipped.at_maybe(1).value_or(2),
                                    flipped.at_maybe(2).value_or(3),
                                    flipped.at_maybe(3).value_or(3));
}

cudnnDataType_t ff_to_cudnn_datatype(DataType type) {
  switch (type) {
    case DataType::FLOAT:
      return CUDNN_DATA_FLOAT;
    case DataType::DOUBLE:
      return CUDNN_DATA_DOUBLE;
    case DataType::INT32:
      return CUDNN_DATA_INT32;
    default:
      assert(false && "Unsupported cudnn data type");
  }
  return CUDNN_DATA_FLOAT;
}

cudaDataType_t ff_to_cuda_datatype(DataType type) {
  switch (type) {
    case DataType::FLOAT:
      return CUDA_R_32F;
    case DataType::DOUBLE:
      return CUDA_R_64F;
    case DataType::INT32:
      return CUDA_R_32I;
    default:
      assert(false && "Unspoorted cuda data type");
  }
  return CUDA_R_32F;
}

template __global__ void
    assign_kernel<half>(half *ptr, size_t size, half value);
template __global__ void
    assign_kernel<float>(float *ptr, size_t size, float value);
template __global__ void
    assign_kernel<double>(double *ptr, size_t size, double value);
template __global__ void
    assign_kernel<int32_t>(int32_t *ptr, size_t size, int32_t value);
template __global__ void
    assign_kernel<int64_t>(int64_t *ptr, size_t size, int64_t value);

template __global__ void
    add_kernel<float>(float *dst, float const *src, size_t size);
template __global__ void
    add_kernel<double>(double *dst, double const *src, size_t size);
template __global__ void
    add_kernel<int32_t>(int32_t *dst, int32_t const *src, size_t size);
template __global__ void
    add_kernel<int64_t>(int64_t *dst, int64_t const *src, size_t size);
template __global__ void
    add_kernel<bool>(bool *dst, bool const *src, unsigned long size);

template __global__ void
    copy_kernel<float>(float *dst, float const *src, coord_t size);
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

template __global__ void apply_add_with_scale<bool>(bool *data_ptr,
                                                    bool const *grad_ptr,
                                                    unsigned long size,
                                                    bool scale);

template __host__ void
    print_tensor<float>(float const *ptr, size_t rect, char const *prefix);
template __host__ void
    print_tensor<double>(double const *ptr, size_t rect, char const *prefix);
template __host__ void
    print_tensor<int32_t>(int32_t const *ptr, size_t rect, char const *prefix);
template __host__ void
    print_tensor<int64_t>(int64_t const *ptr, size_t rect, char const *prefix);
