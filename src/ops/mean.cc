/* Copyright 2021 CMU
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

#include "flexflow/ops/mean.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::coord_t;
using Legion::PhysicalRegion;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;

Tensor FFModel::mean(const Tensor input,
                     const std::vector<int>& dims,
                     bool keepdims,
                     const char *name)
{
  assert(false);
#ifdef DEADCODE
  Mean *mean = new Mean(*this, input, dims, keepdims, name);
  layers.push_back(mean);
  return mean->outputs[0];
#endif
}

Mean::Mean(FFModel& model,
           const ParallelTensor input,
           const std::vector<int>& reduce_dims,
           bool keepdims,
           const char *name)
: Op(model, OP_REDUCE_MEAN, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, input)
{
  //TODO: switch to use the Legion dim ordering
  ParallelDim dims[MAX_TENSOR_DIM];
  int num_dim = 0;
  for (int i = 0; i < inputs[0]->num_dims; i++) {
    bool reduce_this_dim = false;
    for (const auto& dim : reduce_dims)
      if (inputs[0]->num_dims - 1 - dim == i)
        reduce_this_dim = true;
    if (!reduce_this_dim) {
      dims[num_dim++] = inputs[0]->dims[i];
    } else if (keepdims) {
      dims[num_dim++] = inputs[0]->dims[i];
      dims[num_dim-1].size = 1;
    }
  }
  numOutputs = 1;
  numWeights = 0;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      num_dim, dims, input->data_type, this);
}

void Mean::init(const FFModel& ff)
{
}

OpMeta* Mean::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  FFHandler handler = *((const FFHandler*) task->local_args);
  OpMeta* m = new OpMeta(handler);
  return m;
}

void Mean::forward(const FFModel& ff)
{}

void Mean::forward_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{}

void Mean::backward(const FFModel& ff)
{}

void Mean::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{}

bool Mean::measure_operator_cost(
    Simulator* sim,
    const MachineView& mv,
    CostMetrics& cost_metrics) const
{
  return false;
}

}; // namespace FlexFlow
