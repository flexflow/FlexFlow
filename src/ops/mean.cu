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

#include "model.h"
#include "cuda_helper.h"

Tensor FFModel::mean(const Tensor& input,
                     const std::vector<int>& dims,
                     bool keepdims,
                     const char *name)
{
  Mean *mean = new Mean(*this, input, dims, keepdims, name);
  layers.push_back(mean);
  return mean->outputs[0];
}

Mean::Mean(FFModel& model,
           const Tensor& input,
           const std::vector<int>& dims,
           bool keepdims,
           const char *name)
: Op(model, OP_REDUCE_MEAN, name, input)
{
  //TODO: switch to use the Legion dim ordering
  outputs[0].numDim = 0;
  int num_dim = inputs[0].numDim;
  for (int i = 0; i < num_dim; i++) {
    bool reduce_this_dim = false;
    for (const auto& dim : dims)
      if (num_dim - 1 - dim == i)
        reduce_this_dim = true;
    if (!reduce_this_dim) {
      outputs[0].adim[outputs[0].numDim++] = inputs[0].adim[i];
    } else if (keepdims) {
      outputs[0].adim[outputs[0].numDim++] = 1;
    }
  }
  numOutputs = 1;
  numWeights = 0;
}

void Mean::create_weights(FFModel& model)
{
  // DO nothing
}

void Mean::create_output_and_partition(FFModel& model)
{}

OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{}

void Mean::init(const FFModel& ff)
{
}

void Mean::forward_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{}

void Mean::forward(const FFModel& ff)
{}

void Mean::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{}

void Mean::backward(const FFModel& ff)
{}

bool Mean::measure_operator_cost(Simulator* sim,
                                 const ParallelConfig& pc,
                                 CostMetrics& cost_metrics)
{
  return false;
}

