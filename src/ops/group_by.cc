/* Copyright 2019 Stanford
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

#include "model.h"

void group_by_forward(const float* input,
        const int* exp_assign,
        float* output,
        int n, // num experts
        int k, // chosen experts
        float alpha, // factor additional memory assigned
        int batch_size,
        int data_dim)
{
  std::vector<int> expert_idx(n, 0);
  int exp_tensor_rows = alpha*k/n*batch_size;
  int exp_tensor_entries = exp_tensor_rows * data_dim;

  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < k; j++) {
      int expert = exp_assign[i*k + j];
      int row = expert_idx[expert];

      if(row >= exp_tensor_rows)
        continue;

      // copy over sample (maybe better memcpy or so)
      int output_start = expert*exp_tensor_entries + row*data_dim;
      int input start = i*data_dim;
      for(int l = 0; l < data_dim; l++) {
        output[output_start + l] = input[input start + l];
      }
      expert_idx[expert]++;
    }
  }
}



void group_by_backward(const int64_t* input,
		                const int* lengths,
                    const float* output,
                    float* embed,
                    int block_size,
                    int output_size,
                    int index_size,
                    int data_size)
{
  // TODO: Do nothing? - don't propagate expert gradients further
}


void Group_by::forward_task_cpu(const Task *task,
                                const std::vector<PhysicalRegion>& regions,
                                Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);

  // get regions
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA); // TODO: We can make this 3D, more elegant
  const AccessorRO<int, 2> acc_assign(regions[2], FID_DATA);

  Rect<2> rect_data = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  // get further parameters
  // TODO: n, k, alpha

  // Get data shapes and rnsure they match
  /* TODO: I think dimension indices should be other way around */
  coord_t input_rows = rect_input.hi[0] - rect_input.lo[0] + 1;
  coord_t output_rows = rect_output.hi[0] - rect_output.lo[0] + 1;
  int batch_size = input_rows;
  assert((int)(alpha*k*(int)output_rows) == batch_size);

  coord_t input_cols = rect_input.hi[1] - rect_input.lo[1] + 1;
  assert(input_cols == rect_output.hi[1] - rect_output.lo[1] + 1);
  int data_dim = input_cols;

  assert(input_rows == rect_assign.hi[0] - rect_assign.lo[0] + 1);
  assert(k == (int)(rect_assign.hi[1] - rect_assign.lo[1] + 1));

  group_by_forward(acc_input.ptr(rect_input), acc_assign.ptr(rect_assign),
      acc_output.ptr(rect_output), n, k, alpha, batch_size, data_dim);
}


void Group_by::backward_task_cpu(const Task *task,
                                  const std::vector<PhysicalRegion>& regions,
                                  Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);

  // get regions TODO: Indices etc.
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRO<int, 2> acc_assign(regions[2], FID_DATA);

  Rect<2> rect_data = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  // get further parameters
  // TODO: n, k, alpha

  // Get data shapes and rnsure they match
  /* TODO: Indices in embedding.cc are other way around */
  coord_t input_rows = rect_input.hi[0] - rect_input.lo[0] + 1;
  coord_t output_rows = rect_output.hi[0] - rect_output.lo[0] + 1;
  int batch_size = input_rows;
  assert((int)(alpha*k*(int)output_rows) == batch_size);

  coord_t input_cols = rect_input.hi[1] - rect_input.lo[1] + 1;
  assert(input_cols == rect_output.hi[1] - rect_output.lo[1] + 1);
  int data_dim = input_cols;

  assert(input_rows == rect_assign.hi[0] - rect_assign.lo[0] + 1);
  assert(k == (int)(rect_assign.hi[1] - rect_assign.lo[1] + 1));

  group_by_backward(acc_input.ptr(rect_input), acc_assign.ptr(rect_assign),
      acc_output.ptr(rect_output), n, k, alpha, batch_size, data_dim);
}
