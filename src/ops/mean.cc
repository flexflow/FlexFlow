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

#include "flexflow/ops/mean.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

Tensor FFModel::mean(const Tensor input,
                     std::vector<int> const &dims,
                     bool keepdims,
                     char const *name) {
  assert(false);
#ifdef DEADCODE
  Mean *mean = new Mean(*this, input, dims, keepdims, name);
  layers.push_back(mean);
  return mean->outputs[0];
#endif
}

Mean::Mean(FFModel &model,
           const ParallelTensor input,
           std::vector<int> const &reduce_dims,
           bool keepdims,
           char const *name)
    : Op(model,
         OP_REDUCE_MEAN,
         input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         input) {
  // TODO: switch to use the Legion dim ordering
  ParallelDim dims[MAX_TENSOR_DIM];
  int num_dim = 0;
  for (int i = 0; i < inputs[0]->num_dims; i++) {
    bool reduce_this_dim = false;
    for (auto const &dim : reduce_dims) {
      if (inputs[0]->num_dims - 1 - dim == i) {
        reduce_this_dim = true;
      }
    }
    if (!reduce_this_dim) {
      dims[num_dim++] = inputs[0]->dims[i];
    } else if (keepdims) {
      dims[num_dim++] = inputs[0]->dims[i];
      dims[num_dim - 1].size = 1;
    }
  }
  numOutputs = 1;
  numWeights = 0;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      num_dim, dims, input->data_type, this);
}

void Mean::init(FFModel const &ff) {}

OpMeta *Mean::init_task(Task const *task,
                        std::vector<PhysicalRegion> const &regions,
                        Context ctx,
                        Runtime *runtime) {
  FFHandler handler = *((FFHandler const *)task->local_args);
  return nullptr;
}

void Mean::forward(FFModel const &ff) {}

void Mean::forward_task(Task const *task,
                        std::vector<PhysicalRegion> const &regions,
                        Context ctx,
                        Runtime *runtime) {}

void Mean::backward(FFModel const &ff) {}

void Mean::backward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {}

bool Mean::measure_operator_cost(Simulator *sim,
                                 MachineView const &mv,
                                 CostMetrics &cost_metrics) const {
  return false;
}

}; // namespace FlexFlow
