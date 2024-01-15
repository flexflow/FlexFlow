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

#include "flexflow/ops/gather.h"
#include "flexflow/ops/embedding.h"
#include "flexflow/ops/kernels/gather_kernels.h"
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
using PCG::Node;

using namespace FlexFlow::Kernels::Gather;

bool operator==(GatherParams const &lhs, GatherParams const &rhs) {
  return lhs.legion_dim == rhs.legion_dim;
}

bool GatherParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const {
  if (!input.first.is_valid()) {
    return false;
  }
  if (!input.second.is_valid()) {
    return false;
  }
  if (input.first.num_dims != input.second.num_dims) {
    return false;
  }
  for (int i = 0; i < input.first.num_dims; i++) {
    if (i != legion_dim &&
        input.first.dims[i].size < input.second.dims[i].size) {
      return false;
    }
  }
  return true;
}

GatherParams Gather::get_params() const {
  GatherParams params;
  params.legion_dim = this->legion_dim;
  params.layer_guid = this->layer_guid;
  return params;
}

Tensor FFModel::gather(const Tensor input,
                       const Tensor index,
                       int dim,
                       char const *name) {
  Layer *gather = new Layer(this,
                            OP_GATHER,
                            DT_FLOAT,
                            name,
                            2 /*inputs*/,
                            0 /*weights*/,
                            1 /*output*/,
                            input,
                            index);
  assert(index->data_type == DT_INT32 || index->data_type == DT_INT64);
  assert(input->num_dims == index->num_dims);
  int legion_dim = input->num_dims - 1 - dim;
  // https://pytorch.org/docs/stable/generated/torch.gather.html
  // Currently we assume index.size(d) == input.size(d) for all
  // dimensions d != dim, which is a stronger constraint that PyTorch's
  for (int i = 0; i < input->num_dims; i++) {
    if (i != legion_dim) {
      assert(input->dims[i] == index->dims[i]);
    }
  }
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < index->num_dims; i++) {
    dims[i] = index->dims[i];
  }
  gather->outputs[0] = create_tensor_legion_ordering(
      index->num_dims, dims, input->data_type, gather, 0, true /*create_grad*/);
  gather->add_int_property("legion_dim", legion_dim);
  layers.push_back(gather);
  return gather->outputs[0];
}

Op *Gather::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("legion_dim", value);
  int legion_dim = value;
  return new Gather(
      model, layer->layer_guid, inputs[0], inputs[1], legion_dim, layer->name);
}

Gather::Gather(FFModel &model,
               GatherParams const &params,
               std::pair<ParallelTensor, ParallelTensor> const &inputs,
               char const *name)
    : Gather(model,
             params.layer_guid,
             inputs.first,
             inputs.second,
             params.legion_dim,
             params.name) {}

Gather::Gather(FFModel &model,
               LayerID const &_layer_guid,
               const ParallelTensor input,
               const ParallelTensor index,
               int _legion_dim,
               char const *name)
    : Op(model,
         OP_GATHER,
         input->data_type,
         name,
         2 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         input,
         index),
      legion_dim(_legion_dim) {
  layer_guid = _layer_guid;
  // Assume that input and index have the same paralleldim except
  // for the legion_dim-th dim, which cannot be parallelized
  for (int i = 0; i < input->num_dims; i++) {
    if (i != legion_dim) {
      assert(input->dims[i] == index->dims[i]);
    }
  }
  assert(index->dims[legion_dim].degree == 1);
  assert(input->dims[legion_dim].degree == 1);
  // output has the same parallel dims as index
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < index->num_dims; i++) {
    dims[i] = index->dims[i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      index->num_dims, dims, input->data_type, this);
}

void Gather::serialize(Legion::Serializer &sez) const {
  GatherParams params = get_params();
  sez.serialize(params.legion_dim);
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

using PCG::Node;
/*static*/
Node Gather::deserialize(FFModel &ff,
                         Legion::Deserializer &dez,
                         ParallelTensor inputs[],
                         int num_inputs) {
  assert(num_inputs == 2);
  int legion_dim;
  dez.deserialize(legion_dim);
  size_t id, transformer_layer_id, deserialized_model_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);

  GatherParams params;
  params.legion_dim = legion_dim;
  params.layer_guid = layer_guid;
  strcpy(params.name, name);
  return ff.get_or_create_node<Gather>({inputs[0], inputs[1]}, params);
}

Op *Gather::materialize(FFModel &ff,
                        ParallelTensor inputs[],
                        int num_inputs) const {
  GatherParams params = get_params();
  return new Gather(ff, params, {inputs[0], inputs[1]}, this->name);
}

void Gather::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(GATHER_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Gather)),
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
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *Gather::init_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  Gather const *gather = (Gather const *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  GatherMeta *m = new GatherMeta(handle, gather);
  std::strcpy(m->op_name, gather->name);
  m->layer_guid = gather->layer_guid;
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR index = helperGetGenericTensorAccessorRO(
      m->input_type[1], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  assert(input.domain.get_dim() == index.domain.get_dim());
  assert(output.domain.get_dim() == index.domain.get_dim());
  for (int i = 0; i < input.domain.get_dim(); i++) {
    assert(index.domain.hi()[i] == output.domain.hi()[i]);
    assert(index.domain.lo()[i] == output.domain.lo()[i]);
    if (i != m->legion_dim) {
      assert(input.domain.hi()[i] == index.domain.hi()[i]);
      assert(input.domain.lo()[i] == index.domain.lo()[i]);
    }
  }
  return m;
}

void Gather::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(GATHER_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, false),
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
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Gather::forward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  GatherMeta const *m = *((GatherMeta **)task->local_args);
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR index = helperGetGenericTensorAccessorRO(
      m->input_type[1], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  forward_kernel_wrapper(m, input, index, output);
}

void Gather::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(GATHER_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Gather::backward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  GatherMeta const *m = *((GatherMeta **)task->local_args);
  GenericTensorAccessorR output_grad = helperGetGenericTensorAccessorRO(
      m->output_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR index = helperGetGenericTensorAccessorRO(
      m->input_type[1], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  backward_kernel_wrapper(m, output_grad, index, input_grad);
}

bool Gather::measure_operator_cost(Simulator *sim,
                                   MachineView const &mv,
                                   CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_input, sub_index, sub_output;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  if (!inputs[1]->get_sub_tensor(mv, sub_index)) {
    return false;
  }
  GatherMeta *m = new GatherMeta(sim->handler, this);
  sim->free_all();
  bool out_of_memory = false;
  Domain input_domain = sub_input.get_domain();
  void *input_ptr = sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  GenericTensorAccessorW input_acc(
      inputs[0]->data_type, input_domain, input_ptr);
  Domain index_domain = sub_index.get_domain();
  void *index_ptr = sim->allocate(sub_index.get_volume(), inputs[1]->data_type);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  GenericTensorAccessorW index_acc(
      inputs[1]->data_type, index_domain, index_ptr);
  out_of_memory = out_of_memory || (input_ptr == NULL) || (index_ptr == NULL);
  Domain out_domain = sub_output.get_domain();
  void *output_ptr =
      sim->allocate(sub_output.get_volume(), outputs[0]->data_type);
  out_of_memory = out_of_memory || (output_ptr == NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  GenericTensorAccessorW output_acc(
      outputs[0]->data_type, out_domain, output_ptr);
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(m, input_acc, index_acc, output_acc);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    backward = [&] {
      backward_kernel_wrapper(m, output_acc, index_acc, input_acc);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Gather] name(%s) forward_time(%.4lf) "
           "backward_time(%.4lf)\n",
           name,
           cost_metrics.forward_time,
           cost_metrics.backward_time);
  } else {
    printf("[Measure Gather] name(%s) forward_time(%.4lf)\n",
           name,
           cost_metrics.forward_time);
  }
  delete m;
  return true;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::GatherParams>::operator()(
    FlexFlow::GatherParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.legion_dim);
  hash_combine(key, params.layer_guid.id);
  return key;
}
}; // namespace std
