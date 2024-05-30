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

#include "flexflow/ops/flat.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/flat_kernels.h"
#include "legion/legion_utilities.h"

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

using namespace FlexFlow::Kernels::Flat;

Tensor FFModel::flat(const Tensor input, char const *name) {
  assert(input->num_dims == 4);
  Layer *flat = new Layer(this,
                          OP_FLAT,
                          DT_FLOAT,
                          name,
                          1 /*inputs*/,
                          0 /*weights*/,
                          1 /*outputs*/,
                          input);
  int dims[MAX_TENSOR_DIM];
  dims[1] = input->dims[3];
  dims[0] = input->dims[2] * input->dims[1] * input->dims[0];
  flat->outputs[0] = create_tensor_legion_ordering(
      2, dims, DT_FLOAT, flat, 0, true /*create_grad*/);
  layers.push_back(flat);
  return flat->outputs[0];
}

Op *Flat::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  return new Flat(model, inputs[0], layer->name);
}

int FlatParams::output_size(ParallelTensorShape const &input,
                            ParallelDim output_dims[MAX_TENSOR_DIM]) const {
  output_dims[FlatOutput::REPLICA].is_replica_dim = true;
  output_dims[FlatOutput::SAMPLE].size = input.dims[FlatInput::SAMPLE].size;
  output_dims[FlatOutput::CHANNEL].size =
      (input.dims[FlatInput::CHANNEL].size *
       input.dims[FlatInput::HEIGHT].size * input.dims[FlatInput::WIDTH].size);

  return FlatOutput::NUMDIM;
}

void FlatParams::solve_dims(ParallelTensorShape const &input,
                            ParallelDim output_dims[MAX_TENSOR_DIM],
                            int *output_ndims) const {
  assert((output_dims == nullptr) == (output_ndims == nullptr));

  std::vector<ParallelDimMappingRecord> mapping;
  Flat::construct_output_mappings(mapping);

  std::vector<ParallelDim *> output_dim_sets;
  if (output_dims != nullptr) {
    *output_ndims = this->output_size(input, output_dims);
    output_dim_sets.push_back(output_dims);
  }

  solve_parallel_dim_mappings(mapping, {input.dims}, {}, output_dim_sets);
}

bool FlatParams::is_valid(ParallelTensorShape const &input) const {
  ParallelTensorShape output_shape;

  this->solve_dims(input, output_shape.dims, &output_shape.num_dims);

  bool is_valid = true;
  is_valid &= input.is_valid();
  is_valid &= output_shape.is_valid();
  is_valid &= (input.dims[FlatInput::WIDTH].degree == 1);

  return is_valid;
}

bool operator==(FlatParams const &, FlatParams const &) {
  // flat doesn't have params to compare
  return true;
}

/*static*/
void Flat::construct_output_mappings(
    std::vector<ParallelDimMappingRecord> &mappings) {
  Op::construct_output_parallel_dims(
      mappings,
      {{FlatInput::REPLICA, MappingOperation::PARTITION, FlatOutput::REPLICA},
       {FlatInput::SAMPLE, MappingOperation::PARTITION, FlatOutput::SAMPLE},
       {FlatInput::CHANNEL, MappingOperation::PARTITION, FlatOutput::CHANNEL}});
}

Flat::Flat(FFModel &model, const ParallelTensor _input, char const *name)
    : Op(model,
         OP_FLAT,
         _input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         _input) {
  assert(_input->num_dims == FlatInput::NUMDIM);

  Flat::construct_output_mappings(*this->parallel_dims_mapping);

  ParallelDim output_dims[MAX_TENSOR_DIM];
  int output_ndims;
  this->get_params().solve_dims(
      this->inputs[0]->get_shape(), output_dims, &output_ndims);

  outputs[0] = model.create_parallel_tensor_legion_ordering(
      output_ndims, output_dims, _input->data_type, this);

  assert(check_output_input_weight_parallel_dims());
}

Flat::Flat(FFModel &model,
           FlatParams const &params,
           const ParallelTensor input,
           char const *name)
    : Flat(model, input, name) {}

void Flat::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(FLAT_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Flat)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *Flat::init_task(Task const *task,
                        std::vector<PhysicalRegion> const &regions,
                        Context ctx,
                        Runtime *runtime) {
  FFHandler handler = *((FFHandler const *)task->local_args);
  FlatMeta *m = new FlatMeta(handler);
  return m;
}

void Flat::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(FLAT_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): output
*/
void Flat::forward_task(Task const *task,
                        std::vector<PhysicalRegion> const &regions,
                        Context ctx,
                        Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorR<float, FlatInput::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, FlatOutput::NUMDIM> acc_output(regions[1],
                                                        task->regions[1],
                                                        FID_DATA,
                                                        ctx,
                                                        runtime,
                                                        false /*readOutput*/);
  assert(acc_input.rect.volume() == acc_output.rect.volume());

  forward_kernel_wrapper(
      acc_input.ptr, acc_output.ptr, acc_input.rect.volume());
  // checkCUDA(cudaDeviceSynchronize());
}

void Flat::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(FLAT_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I/O) : input_grad
  regions[1](I) : output_grad
*/
void Flat::backward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorW<float, FlatInput::NUMDIM> acc_input_grad(regions[0],
                                                           task->regions[0],
                                                           FID_DATA,
                                                           ctx,
                                                           runtime,
                                                           true /*readOutput*/);
  TensorAccessorR<float, FlatOutput::NUMDIM> acc_output_grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  assert(acc_input_grad.rect.volume() == acc_output_grad.rect.volume());

  backward_kernel_wrapper(
      acc_input_grad.ptr, acc_output_grad.ptr, acc_input_grad.rect.volume());
}

Domain Flat::get_input_tensor_shape(ParallelConfig const &pc,
                                    int input_idx,
                                    int part_idx) const {
  assert(input_idx < numInputs);
  assert(pc.nDims == 3);
  assert(pc.dim[0] == 1);
  assert(pc.dim[2] == 1);

  Domain d;
  d.dim = inputs[input_idx]->num_dims;
  for (int i = 0; i < d.dim - 1; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i + d.dim] = inputs[input_idx]->dims[i].size - 1;
  }
  assert(inputs[input_idx]->dims[d.dim - 2].size % pc.num_parts() == 0);
  int dim_size = inputs[input_idx]->dims[d.dim - 2].size / pc.num_parts();
  d.rect_data[d.dim - 2] = part_idx * dim_size;
  d.rect_data[2 * d.dim - 2] = d.rect_data[d.dim - 2] + dim_size - 1;
  return d;
}

void Flat::serialize(Legion::Serializer &sez) const {
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
  return;
}

bool Flat::measure_operator_cost(Simulator *sim,
                                 MachineView const &mv,
                                 CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_input, sub_output;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  size_t num_elements = sub_output.get_volume();

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(input_ptr, output_ptr, num_elements);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr =
        (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *output_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    assert(output_grad_ptr != NULL);
    assert(input_grad_ptr != NULL);
    backward = [=] {
      backward_kernel_wrapper(input_grad_ptr, output_grad_ptr, num_elements);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug(
        "[Measure Flat] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    log_measure.debug("[Measure Flat] name(%s) forward_time(%.4lf)\n",
                      name,
                      cost_metrics.forward_time);
  }

  return true;
}

FlatParams Flat::get_params() const {
  FlatParams params;
  return params;
}

using PCG::Node;
/*static*/
Node Flat::deserialize(FFModel &ff,
                       Legion::Deserializer &dez,
                       ParallelTensor inputs[],
                       int num_inputs) {
  assert(num_inputs == 1);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  return ff.get_or_create_node<Flat>(inputs[0], {});
}

Op *Flat::materialize(FFModel &ff,
                      ParallelTensor inputs[],
                      int num_inputs) const {
  assert(num_inputs == 1);
  return new Flat(ff, inputs[0], this->name);
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::FlatParams>::operator()(
    FlexFlow::FlatParams const &params) const {
  size_t key = 0;
  return hash<int>{}(key);
}
}; // namespace std
