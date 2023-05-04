/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "flexflow/ops/gather.h"
#include "flexflow/ops/kernels/gather_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

GatherMeta::GatherMeta(FFHandler handler, Gather const *gather)
    : OpMeta(handler, gather) {
  legion_dim = gather->legion_dim;
}

namespace Kernels {
namespace Gather {

void forward_kernel_wrapper(GatherMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &index,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  coord_t stride = 1;
  for (int i = 0; i < m->legion_dim; i++) {
    stride *= (output.domain.hi()[i] - output.domain.lo()[i] + 1);
  }
  coord_t output_dim_size =
      output.domain.hi()[m->legion_dim] - output.domain.lo()[m->legion_dim] + 1;
  coord_t input_dim_size =
      input.domain.hi()[m->legion_dim] - input.domain.lo()[m->legion_dim] + 1;
  if (index.data_type == DT_INT32) {
    Internal::forward_kernel(input.get_float_ptr(),
                             index.get_int32_ptr(),
                             output.get_float_ptr(),
                             output.domain.get_volume(),
                             stride,
                             input_dim_size,
                             output_dim_size,
                             stream);
  } else {
    assert(index.data_type == DT_INT64);
    Internal::forward_kernel(input.get_float_ptr(),
                             index.get_int64_ptr(),
                             output.get_float_ptr(),
                             output.domain.get_volume(),
                             stride,
                             input_dim_size,
                             output_dim_size,
                             stream);
  }
}

void backward_kernel_wrapper(GatherMeta const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorR const &index,
                             GenericTensorAccessorW const &input_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  coord_t stride = 1;
  for (int i = 0; i < m->legion_dim; i++) {
    stride *= (output_grad.domain.hi()[i] - output_grad.domain.lo()[i] + 1);
  }
  coord_t output_dim_size = output_grad.domain.hi()[m->legion_dim] -
                            output_grad.domain.lo()[m->legion_dim] + 1;
  coord_t input_dim_size = input_grad.domain.hi()[m->legion_dim] -
                           input_grad.domain.lo()[m->legion_dim] + 1;
  if (index.data_type == DT_INT32) {
    Internal::backward_kernel(output_grad.get_float_ptr(),
                              index.get_int32_ptr(),
                              input_grad.get_float_ptr(),
                              output_grad.domain.get_volume(),
                              stride,
                              input_dim_size,
                              output_dim_size,
                              stream);
  } else {
    assert(index.data_type == DT_INT64);
    Internal::backward_kernel(output_grad.get_float_ptr(),
                              index.get_int64_ptr(),
                              input_grad.get_float_ptr(),
                              output_grad.domain.get_volume(),
                              stride,
                              input_dim_size,
                              output_dim_size,
                              stream);
  }
}

namespace Internal {

template <typename IndexType>
__global__ void gather_forward(float const *input,
                               IndexType const *index,
                               float *output,
                               coord_t output_size,
                               coord_t stride,
                               coord_t input_dim_size,
                               coord_t output_dim_size) {
  CUDA_KERNEL_LOOP(o, output_size) {
    // output tensor shape: [*, output_dim_size, stride]
    // output tensor stride: [output_dim_size * stride, stride, 1]
    // output tensor index: [outter_index, index_2, left_over]
    // input tensor shape: [*, input_dim_size, stride]
    // input tensor stride: [input_dim_size * stride, stride, 1]
    // the index of the corresponding input tensor should be:
    // [outter_index, index[0], left_over]
    // Therefore, input_index = outter_index * (stride * input_dim_size)
    //                        + index[0] * stride + left_over;
    coord_t outter_index = o / (stride * output_dim_size);
    // coord_t index_2 = (o / stride) % dim_size
    coord_t left_over = o % stride;
    coord_t input_idx = outter_index * (stride * input_dim_size) +
                        index[o] * stride + left_over;
    output[o] = input[input_idx];
  }
}

template <typename IndexType>
void forward_kernel(float const *input_ptr,
                    IndexType const *index_ptr,
                    float *output_ptr,
                    coord_t output_size,
                    coord_t stride,
                    coord_t input_dim_size,
                    coord_t output_dim_size,
                    cudaStream_t stream) {
  assert(input_ptr != nullptr);
  assert(index_ptr != nullptr);
  assert(output_ptr != nullptr);
  gather_forward<IndexType>
      <<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
          input_ptr,
          index_ptr,
          output_ptr,
          output_size,
          stride,
          input_dim_size,
          output_dim_size);
}

template <typename IndexType>
__global__ void gather_backward(float const *output_grad,
                                IndexType const *index,
                                float *input_grad,
                                coord_t output_size,
                                coord_t stride,
                                coord_t input_dim_size,
                                coord_t output_dim_size) {
  CUDA_KERNEL_LOOP(o, output_size) {
    // output tensor shape: [*, output_dim_size, stride]
    // output tensor stride: [output_dim_size * stride, stride, 1]
    // output tensor index: [outter_index, index_2, left_over]
    // input tensor shape: [*, input_dim_size, stride]
    // input tensor stride: [input_dim_size * stride, stride, 1]
    // the index of the corresponding input tensor should be:
    // [outter_index, index[0], left_over]
    // Therefore, input_index = outter_index * (stride * input_dim_size)
    //                        + index[0] * stride + left_over;
    coord_t outter_index = o / (stride * output_dim_size);
    // coord_t index_2 = (o / stride) % dim_size
    coord_t left_over = o % stride;
    coord_t input_idx = outter_index * (stride * input_dim_size) +
                        index[o] * stride + left_over;

    atomicAdd(&input_grad[input_idx], output_grad[o]);
  }
}

template <typename IndexType>
void backward_kernel(float const *output_grad_ptr,
                     IndexType const *index_ptr,
                     float *input_grad_ptr,
                     coord_t output_size,
                     coord_t stride,
                     coord_t input_dim_size,
                     coord_t output_dim_size,
                     cudaStream_t stream) {
  assert(output_grad_ptr != nullptr);
  assert(input_grad_ptr != nullptr);
  assert(index_ptr != nullptr);
  gather_backward<IndexType>
      <<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
          output_grad_ptr,
          index_ptr,
          input_grad_ptr,
          output_size,
          stride,
          input_dim_size,
          output_dim_size);
}

} // namespace Internal
} // namespace Gather
} // namespace Kernels

}; // namespace FlexFlow
