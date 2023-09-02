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

#include "flexflow/ops/kernels/concat_kernels.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Rect;

namespace Kernels {
namespace Concat {

void init_meta(ConcatMeta *m, int legion_axis) {
  m->legion_axis = legion_axis;
}

void forward_kernel_wrapper(ConcatMeta const *m,
                            GenericTensorAccessorW const &output,
                            GenericTensorAccessorR const *inputs,
                            int num_inputs,
                            int axis) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }
  Internal::forward_kernel(output, inputs, num_inputs, axis, stream);
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    // print_tensor<4, float>(output - output_blk_size, output_rect,
    // "[Concat:forward:output]"); printf("output_blk_size=%zu\n",
    // output_blk_size); print_tensor<4, float>(inputs[0], input_rect[0],
    // "[Concat:forward:input0]"); print_tensor<4, float>(inputs[1],
    // input_rect[1], "[Concat:forward:input1]");
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    printf("[%s] forward time = %.4f ms\n", m->op_name, elapsed);
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
  }
}

void backward_kernel_wrapper(ConcatMeta const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorW const *input_grads,
                             int num_inputs,
                             int axis) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }
  Internal::backward_kernel(output_grad, input_grads, num_inputs, axis, stream);
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    printf("[%s] forward time = %.4f ms\n", m->op_name, elapsed);
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
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

void forward_kernel(GenericTensorAccessorW const &output,
                    GenericTensorAccessorR const *inputs,
                    int num_inputs,
                    int axis,
                    hipStream_t stream) {
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  switch (output.domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = output.domain;                                            \
    calc_blk_size<DIM>(num_blocks, output_blk_size, rect, axis);               \
    for (int i = 0; i < num_inputs; i++) {                                     \
      rect = inputs[i].domain;                                                 \
      coord_t input_num_blocks = 1;                                            \
      calc_blk_size<DIM>(input_num_blocks, input_blk_sizes[i], rect, axis);    \
      assert(input_num_blocks == num_blocks);                                  \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
  }

  off_t offset = 0;
  for (int i = 0; i < num_inputs; i++) {
    hipLaunchKernelGGL(copy_with_stride,
                       GET_BLOCKS(input_blk_sizes[i] * num_blocks),
                       CUDA_NUM_THREADS,
                       0,
                       stream,
                       output.get_float_ptr() + offset,
                       inputs[i].get_float_ptr(),
                       num_blocks,
                       output_blk_size,
                       input_blk_sizes[i]);
    offset += input_blk_sizes[i];
  }
}

void backward_kernel(GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const *input_grads,
                     int num_inputs,
                     int axis,
                     ffStream_t stream) {
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  switch (output_grad.domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = output_grad.domain;                                       \
    calc_blk_size<DIM>(num_blocks, output_blk_size, rect, axis);               \
    for (int i = 0; i < num_inputs; i++) {                                     \
      rect = input_grads[i].domain;                                            \
      coord_t input_num_blocks = 1;                                            \
      calc_blk_size<DIM>(input_num_blocks, input_blk_sizes[i], rect, axis);    \
      assert(input_num_blocks == num_blocks);                                  \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
  }

  off_t offset = 0;
  for (int i = 0; i < num_inputs; i++) {
    hipLaunchKernelGGL(add_with_stride,
                       GET_BLOCKS(input_blk_sizes[i] * num_blocks),
                       CUDA_NUM_THREADS,
                       0,
                       stream,
                       input_grads[i].get_float_ptr(),
                       output_grad.get_float_ptr() + offset,
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
