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

#include "flexflow/ops/softmax.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/softmax_kernels.h"
#include "flexflow/utils/hash_utils.h"
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

using namespace FlexFlow::Kernels::Softmax;

/* Params */
bool operator==(SoftmaxParams const &lhs, SoftmaxParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.dim == rhs.dim;
}

void Softmax::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(this->dim);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

using PCG::Node;
/*static*/
Node Softmax::deserialize(FFModel &ff,
                          Legion::Deserializer &dez,
                          ParallelTensor inputs[],
                          int num_inputs) {
  assert(num_inputs == 1);
  size_t id, transformer_layer_id, deserialized_model_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);
  int dim;
  dez.deserialize(dim);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);

  SoftmaxParams params;
  params.layer_guid = layer_guid;
  params.dim = dim;
  strcpy(params.name, name);
  return ff.get_or_create_node<Softmax>(inputs[0], params);
}

bool SoftmaxParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

SoftmaxParams Softmax::get_params() const {
  SoftmaxParams params;
  params.layer_guid = this->layer_guid;
  params.dim = this->dim;
  if (strlen(this->name) < MAX_OPNAME) {
    strcpy(params.name, this->name);
  }
  return params;
}

Tensor FFModel::softmax(const Tensor _input,
                        int dim,
                        DataType data_type,
                        char const *name) {
  if (data_type == DT_NONE) {
    data_type = _input->data_type;
  }
  Layer *sm = new Layer(this,
                        OP_SOFTMAX,
                        data_type,
                        name,
                        1 /*inputs*/,
                        0 /*weights*/,
                        1 /*outputs*/,
                        _input);
  int numdims = _input->num_dims;
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdims; i++) {
    dims[i] = _input->dims[i];
  }
  sm->outputs[0] = create_tensor_legion_ordering(
      numdims, dims, data_type, sm, 0, true /*create_grad*/);
  sm->add_int_property("softmax_dim", dim);
  layers.push_back(sm);
  return sm->outputs[0];
}

Op *Softmax::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("softmax_dim", value);
  int dim = (int)value;
  return new Softmax(model,
                     layer->layer_guid,
                     inputs[0],
                     (inputs[0]->num_dims - 1 - dim) % inputs[0]->num_dims,
                     layer->name);
}

Softmax::Softmax(FFModel &model,
                 LayerID const &_layer_guid,
                 const ParallelTensor _input,
                 int _dim,
                 char const *name)
    : Op(model,
         OP_SOFTMAX,
         _input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         _input),
      dim(_dim) {
  // Currently assume we always perform softmax along the inner most dim
  assert(dim == 0);
  layer_guid = _layer_guid;
  ParallelDim dims[MAX_TENSOR_DIM];
  int numdim = _input->num_dims;
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[numdim - 1 - i];
  }
  outputs[0] = model.create_parallel_tensor(numdim, dims, data_type, this);
}

Softmax::Softmax(FFModel &model,
                 SoftmaxParams const &params,
                 const ParallelTensor input,
                 char const *name)
    : Softmax(model, params.layer_guid, input, params.dim, params.name) {}

void Softmax::init_inference(FFModel const &ff,
                             std::vector<ParallelTensor> const &batch_inputs,
                             std::vector<ParallelTensor> const &batch_outputs,
                             MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = batch_outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_init_inference(ff, argmap, batch_outputs[0]);
  IndexLauncher launcher(SOFTMAX_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Softmax)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_DISCARD,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void Softmax::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(SOFTMAX_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Softmax)),
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
                                                    WRITE_DISCARD,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

/*
  regions[0]: input
  regions[1]: output
 */
OpMeta *Softmax::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Softmax const *softmax = (Softmax *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(input_domain == output_domain);
  int ndims = input_domain.get_dim();
  Domain domain;
  for (int i = 0; i < ndims - 1; i++) {
    assert(!softmax->outputs[0]->dims[i].is_replica_dim);
  }
  // Only the outter-most dim can be a replica_dim
  if (softmax->outputs[0]->dims[ndims - 1].is_replica_dim) {
    int replica_degree = softmax->outputs[0]->dims[ndims - 1].size;
    domain.dim = ndims - 1;
    for (int i = 0; i < ndims - 1; i++) {
      domain.rect_data[i] = input_domain.rect_data[i];
      domain.rect_data[i + ndims - 1] = input_domain.rect_data[i + ndims];
    }
    domain.rect_data[2 * ndims - 3] =
        (domain.rect_data[2 * ndims - 3] + 1) * replica_degree - 1;
    assert(domain.get_volume() == input_domain.get_volume());
  } else {
    domain = input_domain;
  }
  SoftmaxMeta *m = new SoftmaxMeta(handle, softmax, domain);
  // checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  std::strcpy(m->op_name, softmax->name);
  m->layer_guid = softmax->layer_guid;
  return m;
}

void Softmax::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(SOFTMAX_FWD_TASK_ID,
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

void Softmax::forward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  SoftmaxMeta const *m = *((SoftmaxMeta **)task->local_args);
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);

  forward_kernel_wrapper(m, input, output);
}

void Softmax::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(SOFTMAX_BWD_TASK_ID,
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

void Softmax::backward_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  SoftmaxMeta const *m = *((SoftmaxMeta **)task->local_args);
  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR output_grad = helperGetGenericTensorAccessorRO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  backward_kernel_wrapper(m, input_grad, output_grad);
}

FutureMap Softmax::inference(FFModel const &ff,
                             BatchConfigFuture const &bc,
                             std::vector<ParallelTensor> const &batch_inputs,
                             std::vector<ParallelTensor> const &batch_outputs,
                             MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = batch_outputs[0]->parallel_is;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  set_argumentmap_for_inference(ff, argmap, batch_outputs[0]);
  size_t machine_view_hash = view->hash();
  /* std::cout << "Softmax op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  IndexLauncher launcher(SOFTMAX_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  // if this is the last operator, we add the region below in order to copy the
  // output to the grad tensor
  assert(ff.config.computationMode == COMP_MODE_INFERENCE);
  int last_op = ff.operators.size() - 1;
  assert(ff.operators[last_op]->op_type == OP_ARGMAX ||
         ff.operators[last_op]->op_type == OP_ARG_TOPK ||
         ff.operators[last_op]->op_type == OP_SAMPLING);
  last_op -= 1;
  while (ff.operators[last_op]->op_type == OP_WEIGHT && last_op > 0) {
    last_op -= 1;
  }
  if (ff.operators[last_op] == this) {
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[0]->part_grad,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[0]->region_grad));
    launcher.add_field(2, FID_DATA);
  }
  return runtime->execute_index_space(ctx, launcher);
}

void Softmax::inference_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  assert(task->regions.size() == regions.size());
  assert(regions.size() == 3 || regions.size() == 2);
  bool is_last_op = (regions.size() == 3);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    return;
  }
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  SoftmaxMeta *m = *((SoftmaxMeta **)task->local_args);
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output_grad;
  if (is_last_op) {
    output_grad = helperGetGenericTensorAccessorWO(m->output_type[0],
                                                   regions[2],
                                                   task->regions[2],
                                                   FID_DATA,
                                                   ctx,
                                                   runtime);
  }
  inference_kernel_wrapper(m, bc, is_last_op, input, output, output_grad);
  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    Softmax::save_inference_tensors_to_file(
        m, shard_id, bc, {input}, {}, {output});
  }
}

FutureMap Softmax::peft_bwd(FFModel const &ff,
                            BatchConfigFuture const &bc,
                            std::vector<ParallelTensor> const &batch_inputs,
                            std::vector<ParallelTensor> const &batch_outputs,
                            MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = batch_outputs[0]->parallel_is;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  set_argumentmap_for_inference(ff, argmap, batch_outputs[0]);
  size_t machine_view_hash = view->hash();
  /* std::cout << "Softmax op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  IndexLauncher launcher(SOFTMAX_PEFT_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  launcher.add_region_requirement(
      RegionRequirement(batch_inputs[0]->part_grad,
                        0 /*projection id*/,
                        reset_input_grads[0] ? WRITE_ONLY : READ_WRITE,
                        EXCLUSIVE,
                        batch_inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(batch_outputs[0]->part_grad,
                        0 /*projection id*/,
                        READ_ONLY,
                        EXCLUSIVE,
                        batch_outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

void Softmax::peft_bwd_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(task->regions.size() == regions.size());
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_active_peft_tokens() == 0) {
    return;
  }
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  SoftmaxMeta *m = *((SoftmaxMeta **)task->local_args);
  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR output_grad = helperGetGenericTensorAccessorRO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  peft_bwd_kernel_wrapper(m, bc, input_grad, output_grad);
  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    Softmax::save_inference_tensors_to_file(
        m, shard_id, bc, {input_grad}, {}, {output_grad}, false);
  }
}

bool Softmax::get_int_parameter(PMParameter para, int *value) const {
  switch (para) {
    case PM_SOFTMAX_DIM:
      *value = dim;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool Softmax::measure_operator_cost(Simulator *sim,
                                    MachineView const &mv,
                                    CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_input;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }

  SoftmaxMeta *m = new SoftmaxMeta(sim->handler, this, sub_output.get_domain());

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  GenericTensorAccessorR input_acc(DT_FLOAT, sub_input.get_domain(), input_ptr);
  assert(input_ptr != NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  GenericTensorAccessorW output_acc(
      DT_FLOAT, sub_output.get_domain(), output_ptr);
  assert(output_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  std::function<void()> forward, backward;
  forward = [&] { forward_kernel_wrapper(m, input_acc, output_acc); };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr =
        (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    GenericTensorAccessorW input_grad_acc(
        DT_FLOAT, sub_input.get_domain(), input_grad_ptr);
    assert(input_grad_ptr != NULL);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *output_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    GenericTensorAccessorW output_grad_acc(
        DT_FLOAT, sub_output.get_domain(), output_grad_ptr);
    assert(output_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);
    backward = [&] {
      backward_kernel_wrapper(m, input_grad_acc, output_grad_acc);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure Softmax] name(%s) num_elements(%zu) "
                      "forward_time(%.4lf) backward_time(%.4lf)\n",
                      name,
                      sub_output.get_volume(),
                      cost_metrics.forward_time,
                      cost_metrics.backward_time);
  } else {
    log_measure.debug(
        "[Measure Softmax] name(%s) num_elements(%zu) forward_time(%.4lf)\n",
        name,
        sub_output.get_volume(),
        cost_metrics.forward_time);
  }
  // Free softmaxmeta
  delete m;
  return true;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::SoftmaxParams>::operator()(
    FlexFlow::SoftmaxParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.dim);
  return key;
}
}; // namespace std
