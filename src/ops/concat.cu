/* Copyright 2017 Stanford, NVIDIA
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

#include "flexflow/ops/concat.h"
#include "flexflow/utils/cuda_helper.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Domain;
using Legion::Rect;

template <int N>
void calc_blk_size(coord_t &num_blocks,
                   coord_t &blk_size,
                   Rect<N> rect,
                   int axis) {
  num_blocks = 1;
  blk_size = 1;
  for (int d = 0; d < N; d++) {
    if (d <= axis)
      blk_size *= (rect.hi[d] - rect.lo[d] + 1);
    else
      num_blocks *= (rect.hi[d] - rect.lo[d] + 1);
  }
}

/*static*/
void Concat::forward_kernel(float *output,
                            float const *const *inputs,
                            int num_inputs,
                            int axis,
                            const Domain &out_domain,
                            const Domain *in_domain,
                            cudaStream_t stream) {
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  switch (out_domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = out_domain;                                               \
    calc_blk_size<DIM>(num_blocks, output_blk_size, rect, axis);               \
    for (int i = 0; i < num_inputs; i++) {                                     \
      rect = in_domain[i];                                                     \
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

  for (int i = 0; i < num_inputs; i++) {
    copy_with_stride<<<GET_BLOCKS(input_blk_sizes[i] * num_blocks),
                       CUDA_NUM_THREADS,
                       0,
                       stream>>>(
        output, inputs[i], num_blocks, output_blk_size, input_blk_sizes[i]);
    // printf("output = %x num_blocks=%d output_blk_size=%d
    // input_blk_size[%d]=%d\n",
    //       output, num_blocks, output_blk_size, i, input_blk_sizes[i]);
    output += input_blk_sizes[i];
  }
}

/*static*/
void Concat::forward_kernel_wrapper(const ConcatMeta *m,
                                    float *output,
                                    float const *const *inputs,
                                    int num_inputs,
                                    int axis,
                                    const Domain &out_domain,
                                    const Domain *in_domain) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Concat::forward_kernel(
      output, inputs, num_inputs, axis, out_domain, in_domain, stream);
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

/*static*/
void Concat::backward_kernel(const float *output_grad,
                             float **input_grads,
                             int num_inputs,
                             int axis,
                             const Domain &out_grad_domain,
                             const Domain *in_grad_domain,
                             cudaStream_t stream) {
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  switch (out_grad_domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = out_grad_domain;                                          \
    calc_blk_size<DIM>(num_blocks, output_blk_size, rect, axis);               \
    for (int i = 0; i < num_inputs; i++) {                                     \
      rect = in_grad_domain[i];                                                \
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

  for (int i = 0; i < num_inputs; i++) {
    add_with_stride<<<GET_BLOCKS(input_blk_sizes[i] * num_blocks),
                      CUDA_NUM_THREADS,
                      0,
                      stream>>>(input_grads[i],
                                output_grad,
                                num_blocks,
                                input_blk_sizes[i],
                                output_blk_size);
    output_grad += input_blk_sizes[i];
  }

  // Rect<2> output_rect(Point<2>(0, 0), Point<2>(output_blk_size-1, batch_size
  // - 1)); Rect<2> input_rect(Point<2>(0, 0), Point<2>(input_blk_sizes[0]-1,
  // batch_size - 1)); print_tensor<2, float>(output_grad - output_blk_size,
  // output_rect, "[Concat:backward:output]"); print_tensor<2,
  // float>(input_grads[0], input_rect, "[Concat:backward:input0]");
}

/*static*/
void Concat::backward_kernel_wrapper(const ConcatMeta *m,
                                     const float *output_grad,
                                     float **input_grads,
                                     int num_inputs,
                                     int axis,
                                     const Domain &out_grad_domain,
                                     const Domain *in_grad_domain) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Concat::backward_kernel(output_grad,
                          input_grads,
                          num_inputs,
                          axis,
                          out_grad_domain,
                          in_grad_domain,
                          stream);
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

}; // namespace FlexFlow
