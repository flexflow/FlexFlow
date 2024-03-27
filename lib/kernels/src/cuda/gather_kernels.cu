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


#include "kernels/datatype_dispatch.h"
#include "kernels/gather_kernels.h"

namespace FlexFlow {

GatherPerDeviceState::GatherPerDeviceState(FFHandler handler)
    : PerDeviceOpState(handler){};

namespace Kernels {
namespace Gather {

template <DataType IndexTxype>
struct ForwardKernel {
  void operator()(cudaStream_t stream,
                  GatherPerDeviceState const *m,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorR const &index,
                  GenericTensorAccessorW const &output,
                  size_t stride,
                  size_t input_dim_size,
                  size_t output_dim_size) {
    /*size_t stride = 1;
    for (int i = 0; i < m->legion_dim; i++) {
      stride *= (output.domain.hi()[i] - output.domain.lo()[i] + 1);
    }
    size_t dim_size =
        output.domain.hi()[m->legion_dim] - output.domain.lo()[m->legion_dim] +
    1;
*/
    gather_forward<IndexType><<<GET_BLOCKS(output.domain.get_volume()),
                                CUDA_NUM_THREADS,
                                0,
                                stream>>>(input.get<DT_FLOAT>(),
                                          index.get<IndexType>(),
                                          output.get<DT_FLOAT>(),
                                          output.domain.get_volume(),
                                          stride,
                                          dim_size);
  }
}

void forward_kernel(cudaStream_t stream,
                    GatherPerDeviceState const *m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output,
                    size_t stride,
                    size_t input_dim_size,
                    size_t output_dim_size) {
  DataTypeDispatch1<ForwardKernel>{}(m->index_data_type,
                                     stream,
                                     m,
                                     input,
                                     index,
                                     output,
                                     stride,
                                     input_dim_size,
                                     output_dim_size);
}

template <DataType IndexType>
struct BackwardKernel {
  void operator()(cudaStream_t stream,
                  GatherPerDeviceState const *m,
                  GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorR const &index,
                  GenericTensorAccessorW const &input_grad,
                  size_t stride,
                  size_t input_dim_size,
                  size_t output_dim_size) {
    /*size_t stride = 1;
    for (int i = 0; i < m->legion_dim; i++) {
      stride *= (output_grad.domain.hi()[i] - output_grad.domain.lo()[i] + 1);
    }
    size_t dim_size = output_grad.domain.hi()[m->legion_dim] -
                      output_grad.domain.lo()[m->legion_dim] + 1;
    */
    gather_backward<IndexType><<<GET_BLOCKS(output_grad.domain.get_volume()),
                                 CUDA_NUM_THREADS,
                                 0,
                                 stream>>>(output_grad.get<DT_FLOAT>(),
                                           index.get<IndexType>(),
                                           input_grad.get<DT_FLOAT>(),
                                           output_grad.domain.get_volume(),
                                           stride,
                                           input_dim_size,
                                           output_dim_size);
  }
}

void backward_kernel(cudaStream_t stream,
                     GatherPerDeviceState const *m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &index,
                     GenericTensorAccessorW const &input_grad,
                     size_t stride,
                     size_t input_dim_size,
                     size_t output_dim_size) {
  DataTypeDispatch1<BackwardKernel>{}(m->index_data_type,
                                      stream,
                                      m,
                                      output_grad,
                                      index,
                                      input_grad,
                                      stride,
                                      input_dim_size,
                                      output_dim_size);
}

template <typename IndexType>
__global__ void gather_forward(float const *input,
                               IndexType const *index,
                               float *output,
                               size_t output_size,
                               size_t stride,
                               size_t input_dim_size,
                               size_t output_dim_size) {
  CUDA_KERNEL_LOOP(o, output_size) {
    // output tensor shape: [*, output_dim_size, stride]
    // output tensor stride: [output_dim_size * stride, stride, 1]
    // output tensor index: [outer_index, index_2, left_over]
    // input tensor shape: [*, input_dim_size, stride]
    // input tensor stride: [input_dim_size * stride, stride, 1]
    // the index of the corresponding input tensor should be:
    // [outer_index, index[0], left_over]
    // Therefore, input_index = outer_index * (stride * input_dim_size)
    //                        + index[0] * stride + left_over;
    size_t outer_index = o / (stride * output_dim_size);
    // coord_t index_2 = (o / stride) % dim_size
    size_t left_over = o % stride;
    size_t input_idx =
        outer_index * (stride * input_dim_size) + index[o] * stride + left_over;
    output[o] = input[input_idx];
  }
}

template <typename IndexType>
__global__ void gather_backward(float const *output_grad,
                                IndexType const *index,
                                float *input_grad,
                                size_t output_size,
                                size_t stride,
                                size_t input_dim_size,
                                size_t output_dim_size) {
  CUDA_KERNEL_LOOP(o, output_size) {
    // output tensor shape: [*, output_dim_size, stride]
    // output tensor stride: [output_dim_size * stride, stride, 1]
    // output tensor index: [outer_index, index_2, left_over]
    // input tensor shape: [*, input_dim_size, stride]
    // input tensor stride: [input_dim_size * stride, stride, 1]
    // the index of the corresponding input tensor should be:
    // [outer_index, index[0], left_over]
    // Therefore, input_index = outer_index * (stride * input_dim_size)
    //                        + index[0] * stride + left_over;
    size_t outer_index = o / (stride * output_dim_size);
    // coord_t index_2 = (o / stride) % dim_size
    size_t left_over = o % stride;
    size_t input_idx =
        outer_index * (stride * input_dim_size) + index[o] * stride + left_over;

    atomicAdd(&input_grad[input_idx], output_grad[o]);
  }
}

} // namespace Gather
} // namespace Kernels
} // namespace FlexFlow
