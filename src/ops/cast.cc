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

#include "flexflow/ops/cast.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/cast_kernels.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

using namespace FlexFlow::Kernels::Cast;

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

Tensor FFModel::cast(const Tensor input, DataType dtype, char const *name) {
  Layer *cast = new Layer(this,
                          OP_CAST,
                          dtype,
                          name,
                          1 /*inputs*/,
                          0 /*weights*/,
                          1 /*outputs*/,
                          input);
  int numdims = input->num_dims;
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdims; i++) {
    dims[i] = input->dims[i];
  }
  cast->outputs[0] = create_tensor_legion_ordering(
      numdims, dims, dtype, cast, 0, true /*create_grad*/);
  cast->add_int_property("dtype", dtype);
  layers.push_back(cast);
  return cast->outputs[0];
}

Op *Cast::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("dtype", value);
  DataType dtype = (DataType)value;
  return new Cast(model, inputs[0], dtype, layer->name);
}

CastParams Cast::get_params() const {
  CastParams params;
  params.dtype = this->outputs[0]->data_type;
  return params;
}

bool CastParams::is_valid(ParallelTensorShape const &input) const {
  bool valid = input.is_valid();
  valid &= (input.dims[input.num_dims - 1].degree == 1);
  return valid;
}

bool operator==(CastParams const &lhs, CastParams const &rhs) {
  return lhs.dtype == rhs.dtype;
}

Cast::Cast(FFModel &model,
           ParallelTensor const &input,
           DataType _dtype,
           char const *name)
    : Op(model,
         OP_CAST,
         _dtype,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         input) {
  numOutputs = 1;
  numWeights = 0;
  int numdim = input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = input->dims[i];
  }
  outputs[0] =
      model.create_parallel_tensor_legion_ordering(numdim, dims, _dtype, this);
}

Cast::Cast(FFModel &model,
           CastParams const &params,
           ParallelTensor const &input,
           char const *name)
    : Cast(model, input, params.dtype, params.name) {}

void Cast::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(CAST_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Cast)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Cast::init_inference(FFModel const &ff,
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

  IndexLauncher launcher(CAST_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Cast)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

OpMeta *Cast::init_task(Task const *task,
                        std::vector<PhysicalRegion> const &regions,
                        Context ctx,
                        Runtime *runtime) {
  Cast *cast = (Cast *)task->args;
  FFHandler handler = *((FFHandler const *)task->local_args);
  CastMeta *m = new CastMeta(handler);
  m->input_data_type = cast->inputs[0]->data_type;
  m->output_data_type = cast->outputs[0]->data_type;
  std::strcpy(m->op_name, cast->name);
  m->layer_guid = cast->layer_guid;
  return m;
}

void Cast::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(CAST_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, false),
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

FutureMap Cast::inference(FFModel const &ff,
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

  IndexLauncher launcher(CAST_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, false),
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
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

template <typename IDT>
void Cast::forward_task_with_1_type(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  CastMeta const *m = *((CastMeta **)task->local_args);
  if (m->output_data_type == DT_FLOAT) {
    Cast::forward_task_with_2_type<IDT, float>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_DOUBLE) {
    Cast::forward_task_with_2_type<IDT, double>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_INT32) {
    Cast::forward_task_with_2_type<IDT, int32_t>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_INT64) {
    Cast::forward_task_with_2_type<IDT, int64_t>(task, regions, ctx, runtime);
  }
}

template <typename IDT, typename ODT>
void Cast::forward_task_with_2_type(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  CastMeta const *m = *((CastMeta **)task->local_args);
  // Domain input_domain = runtime->get_index_space_domain(
  //   ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  const IDT *input_ptr = helperGetTensorPointerRO<IDT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  ODT *output_ptr = helperGetTensorPointerWO<ODT>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  forward_kernel_wrapper<IDT, ODT>(
      m, input_ptr, output_ptr, output_domain.get_volume());
}

void Cast::forward_task(Task const *task,
                        std::vector<PhysicalRegion> const &regions,
                        Context ctx,
                        Runtime *runtime) {
  CastMeta const *m = *((CastMeta **)task->local_args);
  if (m->input_data_type == DT_FLOAT) {
    Cast::forward_task_with_1_type<float>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_DOUBLE) {
    Cast::forward_task_with_1_type<double>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT32) {
    Cast::forward_task_with_1_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT64) {
    Cast::forward_task_with_1_type<int64_t>(task, regions, ctx, runtime);
  }
}

void Cast::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(CAST_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, false),
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
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

template <typename IDT>
void Cast::backward_task_with_1_type(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  CastMeta const *m = *((CastMeta **)task->local_args);
  if (m->input_data_type == DT_FLOAT) {
    Cast::backward_task_with_2_type<IDT, float>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_DOUBLE) {
    Cast::backward_task_with_2_type<IDT, double>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT32) {
    Cast::backward_task_with_2_type<IDT, int32_t>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT64) {
    Cast::backward_task_with_2_type<IDT, int64_t>(task, regions, ctx, runtime);
  }
}

template <typename IDT, typename ODT>
void Cast::backward_task_with_2_type(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  // Domain input_domain = runtime->get_index_space_domain(
  //   ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  const IDT *input_ptr = helperGetTensorPointerRO<IDT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  ODT *output_ptr = helperGetTensorPointerRW<ODT>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  backward_kernel_wrapper<IDT, ODT>(
      input_ptr, output_ptr, output_domain.get_volume());
}

void Cast::backward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  CastMeta const *m = *((CastMeta **)task->local_args);
  if (m->output_data_type == DT_FLOAT) {
    Cast::backward_task_with_1_type<float>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_DOUBLE) {
    Cast::backward_task_with_1_type<double>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_INT32) {
    Cast::backward_task_with_1_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_INT64) {
    Cast::backward_task_with_1_type<int64_t>(task, regions, ctx, runtime);
  }
}

bool Cast::measure_operator_cost(Simulator *sim,
                                 MachineView const &mv,
                                 CostMetrics &cost_metrics) const {
  // Assume cast has no cost
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.inputs_memory = 0;
  cost_metrics.outputs_memory = 0;
  cost_metrics.weights_memory = 0;
  return true;
}

void Cast::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->outputs[0]->data_type);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

using PCG::Node;

Node Cast::deserialize(FFModel &ff,
                       Legion::Deserializer &dez,
                       ParallelTensor inputs[],
                       int num_inputs) {
  assert(num_inputs == 1);
  DataType dtype;
  dez.deserialize(dtype);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  return ff.get_or_create_node<Cast>(inputs[0], {dtype});
}

Op *Cast::materialize(FFModel &ff,
                      ParallelTensor inputs[],
                      int num_inputs) const {
  assert(num_inputs == 1);
  return new Cast(ff, inputs[0], this->outputs[0]->data_type, this->name);
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::CastParams>::operator()(
    FlexFlow::CastParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.dtype);
  return key;
}
}; // namespace std
