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

#include "flexflow/ops/reshape.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/reshape_kernels.h"
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

using namespace FlexFlow::Kernels::Reshape;

/* Params */
bool operator==(ReshapeParams const &lhs, ReshapeParams const &rhs) {
  return lhs.shape == rhs.shape;
}

bool ReshapeParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

Tensor FFModel::reshape(const Tensor input,
                        std::vector<int> const &shape,
                        char const *name) {
  Layer *reshape = new Layer(this,
                             OP_RESHAPE,
                             DT_FLOAT,
                             name,
                             1 /*inputs*/,
                             0 /*weights*/,
                             1 /*outputs*/,
                             input);
  int dims[MAX_TENSOR_DIM];
  int numdim = shape.size();
  for (int i = 0; i < numdim; i++) {
    assert(shape[i] > 0);
    dims[i] = shape[i];
  }
  reshape->outputs[0] = create_tensor(
      numdim, dims, input->data_type, reshape, 0, true /*create_grad*/);
  reshape->add_int_vector_property("shape", shape);
  layers.push_back(reshape);
  return reshape->outputs[0];
}

Op *Reshape::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  std::vector<int> shape;
  layer->get_int_vector_property("shape", shape);
  return new Reshape(model, layer->layer_guid, inputs[0], shape, layer->name);
}

Reshape::Reshape(FFModel &model,
                 LayerID const &_layer_guid,
                 const ParallelTensor input,
                 std::vector<int> const &_shape,
                 char const *name)
    : Op(model,
         OP_RESHAPE,
         input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         input) {
  layer_guid = _layer_guid;
  shape_length = _shape.size();
  assert(shape_length <= MAX_TENSOR_DIM);
  for (int i = 0; i < shape_length; i++) {
    shape_array[i] = _shape[i];
  }
  numOutputs = 1;
  numWeights = 0;
  int num_replica_dims = 0;
  for (int i = 0; i < input->num_dims; i++) {
    if (input->dims[i].is_replica_dim) {
      num_replica_dims++;
    }
  }
  // assert that all replica dims are leading dims
  for (int i = 0; i < num_replica_dims; i++) {
    assert(input->dims[input->num_dims - 1 - i].is_replica_dim);
  }
  int numdim = (int)_shape.size();
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i].size = _shape[numdim - 1 - i];
    dims[i].degree = 1;
    dims[i].parallel_idx = -1;
    dims[i].is_replica_dim = false;
  }
  // copy all replica dims
  for (int i = 0; i < num_replica_dims; i++) {
    dims[i + numdim] = input->dims[input->num_dims - 1 - i];
  }
  numdim += num_replica_dims;
  for (int i = num_replica_dims; i < numdim && i < input->num_dims; i++) {
    if (dims[numdim - 1 - i].size !=
        input->dims[input->num_dims - 1 - i].size) {
      break;
    }
    dims[numdim - 1 - i] = input->dims[input->num_dims - 1 - i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, input->data_type, this);
  assert(outputs[0]->get_volume() == inputs[0]->get_volume());
}

Reshape::Reshape(FFModel &model,
                 ReshapeParams const &params,
                 const ParallelTensor input,
                 char const *name)
    : Reshape(model, params.layer_guid, input, params.shape, params.name) {}

void Reshape::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(RESHAPE_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Reshape)),
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

OpMeta *Reshape::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  Reshape const *reshape = (Reshape *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  ReshapeMeta *m = new ReshapeMeta(handle, reshape);
  std::strcpy(m->op_name, reshape->name);
  m->layer_guid = reshape->layer_guid;
  m->data_type = reshape->outputs[0]->data_type;
  return m;
}

void Reshape::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(RESHAPE_FWD_TASK_ID,
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

void Reshape::forward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  // const Reshape* reshape = (const Reshape*) task->args;
  ReshapeMeta const *m = *((ReshapeMeta **)task->local_args);
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(in_domain.get_volume() == out_domain.get_volume());

  if (m->data_type == DT_FLOAT) {
    float const *in_ptr = helperGetTensorPointerRO<float>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    float *out_ptr = helperGetTensorPointerWO<float>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    forward_kernel_wrapper<float>(in_ptr, out_ptr, in_domain.get_volume());
  } else if (m->data_type == DT_DOUBLE) {
    double const *in_ptr = helperGetTensorPointerRO<double>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    double *out_ptr = helperGetTensorPointerWO<double>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    forward_kernel_wrapper<double>(in_ptr, out_ptr, in_domain.get_volume());
  } else if (m->data_type == DT_INT32) {
    int32_t const *in_ptr = helperGetTensorPointerRO<int32_t>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    int32_t *out_ptr = helperGetTensorPointerWO<int32_t>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    forward_kernel_wrapper<int32_t>(in_ptr, out_ptr, in_domain.get_volume());
  } else if (m->data_type == DT_INT64) {
    int64_t const *in_ptr = helperGetTensorPointerRO<int64_t>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    int64_t *out_ptr = helperGetTensorPointerWO<int64_t>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    forward_kernel_wrapper<int64_t>(in_ptr, out_ptr, in_domain.get_volume());
  } else {
    assert(false && "Unsupported data type in Reshape forward");
  }
}

void Reshape::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(RESHAPE_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): output_grad
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[3](I/O): input0_grad
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

ReshapeParams Reshape::get_params() const {
  std::vector<int> shape_vec;
  for (size_t i = 0; i < shape_length; i++) {
    shape_vec.push_back(shape_array[i]);
  }
  ReshapeParams params;
  params.shape = shape_vec;
  params.layer_guid = this->layer_guid;
  if (strlen(this->name) < MAX_OPNAME) {
    strcpy(params.name, this->name);
  }
  return params;
}

void Reshape::backward_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  // const Reshape* reshape = (const Reshape*) task->args;
  ReshapeMeta const *m = *((ReshapeMeta **)task->local_args);
  Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain in_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(in_grad_domain.get_volume() == out_grad_domain.get_volume());

  if (m->data_type == DT_FLOAT) {
    float const *out_grad_ptr = helperGetTensorPointerRO<float>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    float *in_grad_ptr = helperGetTensorPointerRW<float>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    backward_kernel_wrapper<float>(
        in_grad_ptr, out_grad_ptr, in_grad_domain.get_volume());
  } else if (m->data_type == DT_DOUBLE) {
    double const *out_grad_ptr = helperGetTensorPointerRO<double>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    double *in_grad_ptr = helperGetTensorPointerRW<double>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    backward_kernel_wrapper<double>(
        in_grad_ptr, out_grad_ptr, in_grad_domain.get_volume());
  } else if (m->data_type == DT_INT32) {
    int32_t const *out_grad_ptr = helperGetTensorPointerRO<int32_t>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    int32_t *in_grad_ptr = helperGetTensorPointerRW<int32_t>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    backward_kernel_wrapper<int32_t>(
        in_grad_ptr, out_grad_ptr, in_grad_domain.get_volume());
  } else if (m->data_type == DT_INT64) {
    int64_t const *out_grad_ptr = helperGetTensorPointerRO<int64_t>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    int64_t *in_grad_ptr = helperGetTensorPointerRW<int64_t>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    backward_kernel_wrapper<int64_t>(
        in_grad_ptr, out_grad_ptr, in_grad_domain.get_volume());
  } else {
    assert(false && "Unsupported data type in Reshape backward");
  }
}

bool Reshape::measure_operator_cost(Simulator *sim,
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

  assert(sub_output.get_volume() == sub_input.get_volume());
  size_t num_elements = sub_input.get_volume();

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(input_ptr, output_ptr, num_elements);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr =
        (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(input_grad_ptr != NULL);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *output_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(output_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    backward = [&] {
      backward_kernel_wrapper(input_grad_ptr, output_grad_ptr, num_elements);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf(
        "[Measure Reshape] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Reshape] name(%s) forward_time(%.4lf)\n",
           name,
           cost_metrics.forward_time);
  }
  return true;
}

void Reshape::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->shape_length);
  for (size_t i = 0; i < this->shape_length; i++) {
    sez.serialize(this->shape_array[i]);
  }
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

using PCG::Node;

Node Reshape::deserialize(FFModel &ff,
                          Legion::Deserializer &dez,
                          ParallelTensor inputs[],
                          int num_inputs) {
  assert(num_inputs == 1);
  size_t shape_length;
  std::vector<int> shape;
  dez.deserialize(shape_length);
  for (size_t i = 0; i < shape_length; i++) {
    int value;
    dez.deserialize(value);
    shape.push_back(value);
  }
  size_t id, transformer_layer_id, deserialized_model_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);

  ReshapeParams params;
  params.shape = shape;
  params.layer_guid = layer_guid;
  strcpy(params.name, name);
  return ff.get_or_create_node<Reshape>(inputs[0], params);
}

Op *Reshape::materialize(FFModel &ff,
                         ParallelTensor inputs[],
                         int num_inputs) const {
  assert(num_inputs == 1);
  std::vector<int> shape;
  for (size_t i = 0; i < this->shape_length; i++) {
    shape.push_back(shape_array[i]);
  }
  return new Reshape(ff, this->layer_guid, inputs[0], shape, this->name);
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ReshapeParams>::operator()(
    FlexFlow::ReshapeParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.shape.size());
  for (int n : params.shape) {
    hash_combine(key, n);
  }
  hash_combine(key, params.layer_guid.id);
  return key;
}
}; // namespace std
