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
{
  int odim = outputs[0].numDim;
  int idim = inputs[0].numDim;
  switch (odim * MAX_TENSOR_DIM + idim) {
#define DIMFUNC(ODIM, IDIM) \
    case ODIM * MAX_TENSOR_DIM + IDIM: \
    { \
      task_is = model.get_or_create_task_is(ODIM, name); \
      create_output_and_partition_with_dim<ODIM, IDIM>(model); \
      break; \
    }
    LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC
    default:
    {
      // Unsupported dim for Mean operator
      assert(false);
    }
  }
}

template<int ODIM, int IDIM>
void Mean::create_output_and_partition_with_dim(FFModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<ODIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // Create output tensor
  {
    int dims[ODIM];
    for (int i = 0; i < ODIM; i++)
      dims[i] = outputs[0].adim[ODIM-1-i];
    outputs[0] = model.create_tensor<ODIM>(dims, DT_FLOAT, this);
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }
  model.create_data_parallel_partition_with_diff_dims<IDIM, ODIM>(
      inputs[0], (IndexSpaceT<ODIM>)task_is, input_lps[0], input_grad_lps[0]);
}

OpMeta* Mean::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  FFHandler handler = *((const FFHandler*) task->local_args);
  OpMeta* m = new OpMeta(handler);
  return m;
}

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

