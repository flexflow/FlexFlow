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

#include "kernels/concat_kernels.h"
#include "kernels/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Rect;

namespace Kernels {
namespace Concat {

void init_meta(ConcatMeta *m, int legion_axis) {
  m->legion_axis = legion_axis;
}

template <int DIM>
void forward_kernel_wrapper(ConcatMeta const *m,
                            float *output,
                            float const * const *inputs,
                            Legion::Rect<DIM> output_domain,
                            Legion::Rect<DIM> const *input_domains,
                            int num_inputs,
                            int axis) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Internal::forward_kernel<DIM>(output, inputs, output_domain, input_domains, num_inputs, axis);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<4, float>(output - output_blk_size, output_rect,
    // "[Concat:forward:output]"); printf("output_blk_size=%zu\n",
    // output_blk_size); print_tensor<4, float>(inputs[0], input_rect[0],
    // "[Concat:forward:input0]"); print_tensor<4, float>(inputs[1],
    // input_rect[1], "[Concat:forward:input1]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    printf("[%s] forward time = %.4f ms\n", m->op_name, elapsed);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
  }
}

template <int DIM>
void backward_kernel_wrapper(ConcatMeta const *m,
                             float const *output_grad,
                             float * const *input_grads,
                             Legion::Rect<DIM> output_domain,
                             Legion::Rect<DIM> const *input_domains,
                             int num_inputs,
                             int axis) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Internal::backward_kernel(output_grad, input_grads, output_domain, input_domains);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    printf("[%s] forward time = %.4f ms\n", m->op_name, elapsed);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
  }
}

namespace Internal {

template <int N>
void calc_blk_size(coord_t &num_blocks,
                   coord_t &blk_size,
                   Rect<N> rect,
                   int axis) {
  num_blocks = 1;
  blk_size = 1;
  for (int d = 0; d < N; d++) {
    if (d <= axis) {
      blk_size *= (rect.hi[d] - rect.lo[d] + 1);
    } else {
      num_blocks *= (rect.hi[d] - rect.lo[d] + 1);
    }
  }
}

template <int DIM>
void forward_kernel(float *output,
                    float const **inputs,
                    Rect<DIM> output_domain,
                    Rect<DIM> const *input_domains,
                    int num_inputs,
                    int axis,
                    cudaStream_t stream) {
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  Rect<DIM> rect = output_domain;
  calc_blk_size<DIM>(num_blocks, output_blk_size, rect, axis);
  for (int i = 0; i < num_inputs; i++) {
    rect = input_domains[i];
    coord_t input_num_blocks = 1;
    calc_blk_size<DIM>(input_num_blocks, input_blk_sizes[i], rect, axis);
    assert(input_num_blocks == num_blocks);
  }                                                                          

  off_t offset = 0;
  for (int i = 0; i < num_inputs; i++) {
    copy_with_stride<<<GET_BLOCKS(input_blk_sizes[i] * num_blocks),
                       CUDA_NUM_THREADS,
                       0,
                       stream>>>(output + offset,
                                 inputs[i],
                                 num_blocks,
                                 output_blk_size,
                                 input_blk_sizes[i]);
    // printf("output = %x num_blocks=%d output_blk_size=%d
    // input_blk_size[%d]=%d\n",
    //        output, num_blocks, output_blk_size, i, input_blk_sizes[i]);
    offset += input_blk_sizes[i];
  }
}

template <int DIM>
void backward_kernel(float const *output_grad,
                     float * const *input_grads,
                     Rect<DIM> output_domain,
                     Rect<DIM> const *input_domains,
                     int num_inputs,
                     int axis,
                     cudaStream_t stream) {
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  Rect<DIM> rect = output_domain;
  calc_blk_size<DIM>(num_blocks, output_blk_size, rect, axis);
  for (int i = 0; i < num_inputs; i++) {
    rect = input_domains[i];
    coord_t input_num_blocks = 1;
    calc_blk_size<DIM>(input_num_blocks, input_blk_sizes[i], rect, axis);
    assert(input_num_blocks == num_blocks);
  }

  off_t offset = 0;
  for (int i = 0; i < num_inputs; i++) {
    add_with_stride<<<GET_BLOCKS(input_blk_sizes[i] * num_blocks),
                      CUDA_NUM_THREADS,
                      0,
                      stream>>>(input_grads[i],
                                output_grad + offset,
                                num_blocks,
                                input_blk_sizes[i],
                                output_blk_size);
    offset += input_blk_sizes[i];
  }

  // Rect<2> output_rect(Point<2>(0, 0), Point<2>(output_blk_size-1, batch_size
  // - 1)); Rect<2> input_rect(Point<2>(0, 0), Point<2>(input_blk_sizes[0]-1,
  // batch_size - 1)); print_tensor<2, float>(output_grad - output_blk_size,
  // output_rect, "[Concat:backward:output]"); print_tensor<2,
  // float>(input_grads[0], input_rect, "[Concat:backward:input0]");
}

} // namespace Internal
} // namespace Concat
} // namespace Kernels
} // namespace FlexFlow
