/* Copyright 2020 Stanford, Facebook
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

using namespace Legion;

Tensor FFModel::reshape(const Tensor input,
                        const std::vector<int>& shape,
                        const char* name)
{
  Reshape* reshape = new Reshape(*this, input, shape, name);
  layers.push_back(reshape);
  return reshape->outputs[0];
}

Reshape::Reshape(FFModel& model,
                 const Tensor input,
                 const std::vector<int>& shape,
                 const char* name)
: Op(model, OP_RESHAPE, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, input)
{
  numOutputs = 1;
  numWeights = 0;
  int numdim = (int)shape.size();
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i].size = shape[numdim-1-i];
  }
  for (int i = 0; i < numdim && i < input->num_dims; i++) {
    if (dims[numdim-1-i].size != input->dims[input->num_dims-1-i].size)
      break;
    dims[numdim-1-i] = input->dims[input->num_dims-1-i];
  }
  outputs[0] = model.create_tensor_legion_ordering(numdim, dims, input->data_type, this);
  assert(outputs[0]->get_volume() == inputs[0]->get_volume());
}

#ifdef DEADCODE
void Reshape::create_input_partition(FFModel& model)
{
  switch(inputs[0]->num_dims) {
    case 1:
    {
      if (outputs[0]->num_dims == 1) {
        create_input_partition_with_dim<1, 1>(model);
      } else if (outputs[0]->num_dims == 2) {
        create_input_partition_with_dim<1, 2>(model);
      } else if (outputs[0]->num_dims == 3) {
        create_input_partition_with_dim<1, 3>(model);
      } else if (outputs[0]->num_dims == 4) {
        create_input_partition_with_dim<1, 4>(model);
#if MAX_TENSOR_DIM >= 5
      } else if (outputs[0]->num_dims == 5) {
        create_input_partition_with_dim<1, 5>(model);
#endif
      } else {
        assert(false);
      }
      break;
    }
    case 2:
    {
      if (outputs[0]->num_dims == 1) {
        create_input_partition_with_dim<2, 1>(model);
      } else if (outputs[0]->num_dims == 2) {
        create_input_partition_with_dim<2, 2>(model);
      } else if (outputs[0]->num_dims == 3) {
        create_input_partition_with_dim<2, 3>(model);
      } else if (outputs[0]->num_dims == 4) {
        create_input_partition_with_dim<2, 4>(model);
#if MAX_TENSOR_DIM >= 5
      } else if (outputs[0]->num_dims == 5) {
        create_input_partition_with_dim<2, 5>(model);
#endif
      } else {
        assert(false);
      }
      break;
    }
    case 3:
    {
      if (outputs[0]->num_dims == 1) {
        create_input_partition_with_dim<3, 1>(model);
      } else if (outputs[0]->num_dims == 2) {
        create_input_partition_with_dim<3, 2>(model);
      } else if (outputs[0]->num_dims == 3) {
        create_input_partition_with_dim<3, 3>(model);
      } else if (outputs[0]->num_dims == 4) {
        create_input_partition_with_dim<3, 4>(model);
#if MAX_TENSOR_DIM >= 5
      } else if (outputs[0]->num_dims == 5) {
        create_input_partition_with_dim<3, 5>(model);
#endif
      } else {
        assert(false);
      }
      break;
    }
    case 4:
    {
      if (outputs[0]->num_dims == 1) {
        create_input_partition_with_dim<4, 1>(model);
      } else if (outputs[0]->num_dims == 2) {
        create_input_partition_with_dim<4, 2>(model);
      } else if (outputs[0]->num_dims == 3) {
        create_input_partition_with_dim<4, 3>(model);
      } else if (outputs[0]->num_dims == 4) {
        create_input_partition_with_dim<4, 4>(model);
#if MAX_TENSOR_DIM >= 5
      } else if (outputs[0]->num_dims == 5) {
        create_input_partition_with_dim<4, 5>(model);
#endif
      } else {
        assert(false);
      }
      break;
    }
#if MAX_TENSOR_DIM >= 5
    case 5:
    {
      if (outputs[0]->num_dims == 1) {
        create_input_partition_with_dim<5, 1>(model);
      } else if (outputs[0]->num_dims == 2) {
        create_input_partition_with_dim<5, 2>(model);
      } else if (outputs[0]->num_dims == 3) {
        create_input_partition_with_dim<5, 3>(model);
      } else if (outputs[0]->num_dims == 4) {
        create_input_partition_with_dim<5, 4>(model);
      } else if (outputs[0]->num_dims == 5) {
        create_input_partition_with_dim<5, 5>(model);
      } else {
        assert(false);
      }
      break;
    }
#endif
    default:
      assert(false);
  }
}

template<int IDIM, int ODIM>
void Reshape::create_input_partition_with_dim(FFModel& model)
{
  task_is = IndexSpaceT<ODIM>(model.get_or_create_task_is(ODIM, name));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<ODIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_tasks = part_rect.volume();
  // Data parallel
  for (int i = 0; i < ODIM - 1; i++)
    assert(part_rect.lo[i] == part_rect.hi[i]);
  // number batches has to be divisible by partitions
  assert(inputs[0]->dims[inputs[0]->num_dims-1].size % num_tasks == 0);
  // Create output tensor
  //int output_shape[ODIM];
  //for (int i = 0; i < ODIM; i++)
  //  output_shape[i] = outputs[0].adim[ODIM-1-i];
  //outputs[0] = model.create_tensor<ODIM>(output_shape, DT_FLOAT, this);
  //outputs[0].owner_op = this;
  //outputs[0].owner_idx = 0;
  model.create_data_parallel_partition_with_diff_dims<IDIM, ODIM>(
      inputs[0], (IndexSpaceT<ODIM>)task_is, input_lps[0], input_grad_lps[0]);
}
#endif

OpMeta* Reshape::init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  return NULL;
}

void Reshape::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(RESHAPE_INIT_TASK_ID, parallel_is,
      TaskArgument(this, sizeof(Reshape)), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*static*/
void Reshape::forward_kernel(const float* input_ptr,
                             float* output_ptr,
                             size_t num_elements)
{
  checkCUDA(cudaMemcpyAsync(output_ptr, input_ptr,
      num_elements * sizeof(float), cudaMemcpyDeviceToDevice));
}

void Reshape::forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Reshape* reshape = (const Reshape*) task->args;
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(in_domain.get_volume() == out_domain.get_volume());
  const float* in_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* out_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  forward_kernel(in_ptr, out_ptr, in_domain.get_volume());
}

void Reshape::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(RESHAPE_FWD_TASK_ID, parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Reshape::backward_kernel(float* input_grad_ptr,
                              const float* output_grad_ptr,
                              size_t num_elements)
{
  float alpha = 1.0f;
  apply_add_with_scale<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(
      input_grad_ptr, output_grad_ptr, num_elements, alpha);

}

void Reshape::backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Reshape* reshape = (const Reshape*) task->args;
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(in_grad_domain.get_volume() == out_grad_domain.get_volume());

  const float* out_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* in_grad_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  backward_kernel(in_grad_ptr, out_grad_ptr, in_grad_domain.get_volume());
}

void Reshape::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(RESHAPE_BWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[3](I/O): input0_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Reshape::measure_operator_cost(Simulator* sim,
                                    const ParallelConfig& pc,
                                    CostMetrics& cost_metrics) const
{
  TensorBase sub_input, sub_output;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input, op_type)) {
    return false;
  }

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);
  assert (sub_output.get_volume() == sub_input.get_volume());
  size_t num_elements = sub_input.get_volume();

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(input_ptr, output_ptr, num_elements);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(input_grad_ptr != NULL);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);

    backward = [&] {
      backward_kernel(input_grad_ptr, output_grad_ptr, num_elements);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Meausre Reshape] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Meausre Reshape] name(%s) forward_time(%.4lf)\n",
        name, cost_metrics.forward_time);
  }
  return true;
}

