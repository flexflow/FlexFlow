/* Copyright 2018 Stanford
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

#include "ops/flat.h"
#include "cuda_helper.h"

using namespace Legion;

Tensor FFModel::flat(const Tensor input,
                     const char* name)
{
  //assert(strategies.find(name) != strategies.end());
  //ParallelConfig pc = strategies[name];
  Flat *flat = new Flat(*this, input, name);
  layers.push_back(flat);
  return flat->outputs[0];
}

namespace Input {
  constexpr int NUMDIM = 5,
                WIDTH = 0,
                HEIGHT = 1,
                CHANNEL = 2,
                SAMPLE = 3,
                REPLICA = 4;
}

namespace Output {
  constexpr int NUMDIM = 3,
                CHANNEL = 0,
                SAMPLE = 1,
                REPLICA = 2;
}

int output_size(const Tensor input, ParallelDim output_dims[MAX_TENSOR_DIM]) {
  output_dims[Output::REPLICA].is_replica_dim = true;
  output_dims[Output::SAMPLE].size = input->dims[Input::SAMPLE].size;
  output_dims[Output::CHANNEL].size = (
      input->dims[Input::CHANNEL].size * input->dims[Input::HEIGHT].size * input->dims[Input::WIDTH].size 
  );

  return Output::NUMDIM;
}


void solve_dims(const Tensor input, 
                ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims) 
{
  assert ((output_dims == nullptr) == (output_ndims == nullptr));

  std::vector<ParallelDimMappingRecord> mapping;
  Flat::construct_output_mappings(mapping);

  std::vector<ParallelDim *> output_dim_sets;
  if (output_dims != nullptr) {
    *output_ndims = output_size(input, output_dims);
    output_dim_sets.push_back(output_dims);
  }

  solve_parallel_dim_mappings(
      mapping,
      {input->dims},
      {},
      output_dim_sets
  );
}

bool is_valid(const Tensor input) 
{
  ParallelDim output_dims[MAX_TENSOR_DIM];
  int output_ndims;

  solve_dims(
      input,
      output_dims, &output_ndims
  );

  bool is_valid = true;
  is_valid &= input->check_valid();
  is_valid &= ParallelDim::dims_are_valid(output_dims, output_ndims);
  is_valid &= (input->dims[Input::WIDTH].degree == 1);
  is_valid &= (input->dims[Input::WIDTH].degree == 1);

  return is_valid;
}

size_t Flat::get_params_hash() const {
  return this->inputs[0]->get_owner_independent_hash();
}

Node FFModel::get_or_create_flat_node(const Tensor input) 
{
  if (!is_valid(input)) {
    return Node::INVALID_NODE;
  }

  size_t hash = input->get_owner_independent_hash();

  Flat *flat;

  const auto &it = this->cached_flat_ops.find(hash);
  if (it != cached_flat_ops.end()) {
    flat = it->second;
  } else {
    flat = new Flat(*this, input, NULL);
    cached_flat_ops[hash] = flat;
  }

  return this->new_node(flat);
}

/*static*/
void Flat::construct_output_mappings(std::vector<ParallelDimMappingRecord>& mappings) {
  Op::construct_output_parallel_dims(
    mappings,
    {
      { Input::REPLICA, MappingOperation::PARTITION, Output::REPLICA },
      { Input::SAMPLE, MappingOperation::PARTITION, Output::SAMPLE },
      { Input::CHANNEL, MappingOperation::PARTITION, Output::CHANNEL }
    }
  );
}

Flat::Flat(FFModel& model,
           const Tensor _input,
           const char* name)
: Op(model, OP_FLAT, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, _input)
{
  assert(_input->num_dims == Input::NUMDIM);

  Flat::construct_output_mappings(*this->parallel_dims_mapping);

  ParallelDim output_dims[MAX_TENSOR_DIM];
  int output_ndims;
  solve_dims(
      this->inputs[0],
      output_dims, &output_ndims
  );

  outputs[0] = model.create_tensor_legion_ordering(output_ndims, output_dims, _input->data_type, this);

  assert(check_output_input_weight_parallel_dims());
}

OpMeta* Flat::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  FFHandler handler = *((const FFHandler*) task->local_args);
  FlatMeta* m = new FlatMeta(handler);
  return m;
}

void Flat::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(FLAT_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Flat)), argmap,
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
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

/*static*/
void Flat::forward_kernel(const float* input_ptr,
                          float* output_ptr,
                          size_t num_elements)
{
  checkCUDA(cudaMemcpyAsync(output_ptr, input_ptr,
                            num_elements * sizeof(float),
                            cudaMemcpyDeviceToDevice));
}

/*
  regions[0](I): input
  regions[1](O): output
*/
void Flat::forward_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorR<float, Input::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Output::NUMDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  assert(acc_input.rect.volume() == acc_output.rect.volume());
  forward_kernel(acc_input.ptr, acc_output.ptr, acc_input.rect.volume());
  //checkCUDA(cudaDeviceSynchronize());
}

void Flat::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(FLAT_FWD_TASK_ID, parallel_is,
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

void Flat::backward_kernel(float* input_grad_ptr,
                           const float* output_grad_ptr,
                           size_t num_elements)
{
  add_kernel<float><<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(
      input_grad_ptr, output_grad_ptr, num_elements);
}

/*
  regions[0](I/O) : input_grad
  regions[1](I) : output_grad
*/
void Flat::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorW<float, Input::NUMDIM> acc_input_grad(
    regions[0], task->regions[0], FID_DATA, ctx, runtime,
    true/*readOutput*/);
  TensorAccessorR<float, Output::NUMDIM> acc_output_grad(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  assert(acc_input_grad.rect.volume() == acc_output_grad.rect.volume());
  backward_kernel(acc_input_grad.ptr, acc_output_grad.ptr, acc_input_grad.rect.volume());
  //checkCUDA(cudaMemcpyAsync(acc_input_grad.ptr, acc_output_grad.ptr,
  //                          acc_input_grad.rect.volume() * sizeof(float),
  //                          cudaMemcpyDeviceToDevice));
  //checkCUDA(cudaDeviceSynchronize());
}

void Flat::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(FLAT_BWD_TASK_ID, parallel_is,
    TaskArgument(NULL, 0), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Flat::measure_operator_cost(Simulator* sim,
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
  assert (input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);
  size_t num_elements = sub_output.get_volume();

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(input_ptr, output_ptr, num_elements);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    assert (input_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(input_grad_ptr, output_grad_ptr, num_elements);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Flat] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Flat] name(%s) forward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time);
  }

  return true;
}

Domain Flat::get_input_tensor_shape(const ParallelConfig& pc,
    int input_idx, int part_idx) const
{
  assert(input_idx < numInputs);
  assert(pc.nDims == 3);
  assert(pc.dim[0] == 1);
  assert(pc.dim[2] == 1);

  Domain d;
  d.dim = inputs[input_idx]->num_dims;
  for (int i = 0; i < d.dim-1; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+d.dim] = inputs[input_idx]->dims[i].size - 1;
  }
  assert(inputs[input_idx]->dims[d.dim-2].size % pc.num_parts() == 0);
  int dim_size = inputs[input_idx]->dims[d.dim-2].size / pc.num_parts();
  d.rect_data[d.dim-2] = part_idx * dim_size;
  d.rect_data[2*d.dim-2] = d.rect_data[d.dim-2] + dim_size - 1;
  return d;
}
