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


void agg_function(float* output,
  const float* gate_net_preds,
  const float** exp_preds,
  int k,
  int out_dim)
{
  // NOTE: This is a placeholder, just take first expert's pred
  for(int i = 0; i < out_dim; i++) {
    output[i] = exp_preds[0][i];
  }
}


// TODO: Replace float** exp_preds with correct way of passing
void aggregate_forward(const float** exp_preds,
        const int* exp_assign,
        const float* gating_net_preds,
        float* output,
        int n, // num experts
        int k, // chosen experts
        int batch_size,
        int out_dim)
{
  std:vector<int> expert_idx(n, 0);
  float* chosen_exp_preds[k];
  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < k; j++) {
      // Get pointer to chosen expert predictions
      int expert = exp_assign[k*i + j];
      chosen_exp_preds[j] = exp_preds[expert] + expert_idx[expert]*out_dim;
      expert_idx[expert]++;
    }
    agg_function(output+i*out_dim, gating_net_preds+i*out_dim, chosen_exp_preds,
      k, out_dim);
  }
}


void aggregate_backward()
{
  // TODO
}


void Aggregate::forward_task_cpu(const Task *task,
                                 const std::vector<PhysicalRegion>& regions,
                                 Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);

  // TODO: How is the exp_pred list of tensors passed?
  const AccessorRO<int64_t, 2> acc_assign(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRO<float, 2> acc_gate_preds(regions[2], FID_DATA);

  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_gate_preds = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  // TODO: Get n, k, (alpha). Or we just get it below (lines 96, etc)

  // Get shapes and ensure they match
  // TODO: Indices
  coord_t batch_size = rect_assign.hi[1] - rect_assign.lo[1] + 1;
  assert(rect_output.hi[1] - rect_output.lo[1] + 1 == batch_size);
  assert(rect_gate_preds.hi[1] - rect_gate_preds.lo[1] + 1 == batch_size);
  // NOTE: Each expert pred should have alpha*n/k*batch_size rows

  int k = (int)(rect_assign.hi[0] - rect_assign.lo[0] + 1);
  int n = (int)(rect_gate_preds.hi[0] - rect_gate_preds.lo[0] + 1);
  int out_dim = (int)(rect_output.hi[0] - rect_output.lo[0] + 1);
  // NOTE: Each expert pred should have out_dim columns

  aggregate_forward(exp_preds /*TODO: List of exp_pred s*/, acc_assign.ptr(rect_assign),
    acc_gate_preds.ptr(rect_gate_preds), acc_output.ptr(rect_output), n, k,
    batch_size, out_dim);
}

void Aggregate::backward_task_cpu(const Task *task,
                                  const std::vector<PhysicalRegion>& regions,
                                  Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);

  // TODO: How is the exp_pred list of tensors passed?
  const AccessorRO<int64_t, 2> acc_assign(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRO<float, 2> acc_gate_preds(regions[2], FID_DATA);

  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_gate_preds = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  // TODO: Get n, k, (alpha)

  // Get shapes and ensure they match
  // TODO: Indices
  coord_t batch_size = rect_assign.hi[1] - rect_assign.lo[1] + 1;
  assert(rect_output.hi[1] - rect_output.lo[1] + 1 == batch_size);
  assert(rect_gate_preds.hi[1] - rect_gate_preds.lo[1] + 1 == batch_size);
  // NOTE: Each expert pred should have alpha*n/k*batch_size rows

  int k = (int)(rect_assign.hi[0] - rect_assign.lo[0] + 1);
  int n = (int)(rect_gate_preds.hi[0] - rect_gate_preds.lo[0] + 1);
  int out_dim = (int)(rect_output.hi[0] - rect_output.lo[0] + 1);
  // NOTE: Each expert pred should have out_dim columns

  aggregate_backward(/*TODO:*/);
}
