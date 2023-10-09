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

#include "layer_norm.h"
#include "kernels/layer_norm_kernels.h"
#include "legion/legion_utilities.h"
#include "op-attrs/ops/layer_norm.h"
#include "utils/exception.decl.h"
#include "utils/hash-utils.h"
#include <type_traits>

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

using namespace FlexFlow::Kernels::LayerNorm;

LayerNormParams LayerNorm::get_params() const {
  LayerNormParams params;
  params.layer_guid = this->layer_guid;
  params.axes = this->axes;
  params.elementwise_affine = this->elementwise_affine;
  params.eps = this->eps;
  return params;
}

Tensor FFModel::layer_norm(const Tensor input,
                           std::vector<int> const &axes,
                           bool elementwise_affine,
                           float eps,
                           char const *name) {
  // FIXME: currently disable elementwise_affine
  elementwise_affine = false;
  // axes must be the last axes.size() dimensions
  for (int i = 0; i < axes.size(); i++) {
    bool found = false;
    for (int j = 0; j < axes.size(); j++) {
      if (axes[j] == input->num_dims - 1 - i) {
        found = true;
      }
    }
    if (!found) {
      assert(false && "axes must be the last axes.size() dimensions");
    }
  }
  int num_weights = elementwise_affine ? 2 : 0;
  Layer *ln = new Layer(this,
                        OP_LAYERNORM,
                        DT_FLOAT,
                        name,
                        1 /*inputs*/,
                        num_weights,
                        1 /*outputs*/,
                        input);
  ln->outputs[0] = create_tensor_legion_ordering(input->num_dims,
                                                 input->dims,
                                                 input->data_type,
                                                 ln,
                                                 0,
                                                 true /*create_grad*/);
  if (num_weights == 2) {
    int M = 1;
    for (int i = 0; i < axes.size(); i++) {
      M *= input->dims[input->num_dims - 1 - axes[i]];
    }
    int dims[1] = {M};
    ln->weights[0] = create_weight_legion_ordering(1,
                                                   dims,
                                                   input->data_type,
                                                   ln,
                                                   true /*create_grad*/,
                                                   nullptr,
                                                   CHOSEN_SYNC_TYPE);
    ln->weights[1] = create_weight_legion_ordering(1,
                                                   dims,
                                                   input->data_type,
                                                   ln,
                                                   true /*create_grad*/,
                                                   nullptr,
                                                   CHOSEN_SYNC_TYPE);
  }
  ln->add_int_property("elementwise_affine", elementwise_affine);
  ln->add_int_vector_property("axes", axes);
  ln->add_float_property("eps", eps);
  layers.push_back(ln);
  return ln->outputs[0];
}

Op *LayerNorm::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("elementwise_affine", value);
  bool elementwise_affine = (bool)value;
  std::vector<int> axes;
  layer->get_int_vector_property("axes", axes);
  float eps;
  layer->get_float_property("eps", eps);
  return new LayerNorm(model,
                       layer->layer_guid,
                       inputs[0],
                       axes,
                       elementwise_affine,
                       eps,
                       false, // allocate_weights
                       layer->name);
}

LayerNorm::LayerNorm(FFModel &model,
                     LayerNormParams const &params,
                     ParallelTensor const input,
                     char const *name,
                     bool allocate_weights)
    : LayerNorm(model,
                params.layer_guid,
                input,
                params.axes,
                params.elementwise_affine,
                params.eps,
                allocate_weights,
                name) {}

LayerNorm::LayerNorm(FFModel &model,
                     LayerID const &_layer_guid,
                     const ParallelTensor _input,
                     std::vector<int> const &_axes,
                     bool _elementwise_affine,
                     float _eps,
                     bool allocate_weights,
                     char const *name)
    : Op(model,
         OP_LAYERNORM,
         _input->data_type,
         name,
         1 /*inputs*/,
         _elementwise_affine ? 2 : 0 /*weights*/,
         1 /*outputs*/,
         _input),
      elementwise_affine(_elementwise_affine), eps(_eps), axes(_axes) {
  // overwrite layer_guid
  layer_guid = _layer_guid;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, _input->dims, _input->data_type, this);
  assert(check_output_input_weight_parallel_dims(allocate_weights));
  ParallelDim output_dims[MAX_TENSOR_DIM];
  int M = 1;
  for (int i = 0; i < axes.size(); i++) {
    M *= inputs[0]->dims[inputs[0]->num_dims - 1 - axes[i]].size;
  }
  effective_num_elements = M;
  effective_batch_size = inputs[0]->get_volume() / M;
  if (numWeights > 0 && allocate_weights) {
    int kernel_dims = 2;
    assert(false);
    // weights[0] = model.create_parallel_weight_legion_ordering(
    //     kernel_dims,
  } else {
    // do nothing
  }
  return;
}

void LayerNorm::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(LAYERNORM_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(LayerNorm)),
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

PerDeviceOpState *
    LayerNorm::init_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  LayerNorm *ln = (LayerNorm *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  LayerNormMeta *meta = new LayerNormMeta(handle, ln);
  return meta;
}

void LayerNorm::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(LAYERNORM_FWD_TASK_ID,
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
  if (elementwise_affine) {
    launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      weights[0]->region));
    launcher.add_field(2, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      weights[1]->region));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I/O): gamma
  regions[3](I/O): beta
*/
void LayerNorm::forward_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  LayerNormMeta const *m = *((LayerNormMeta **)task->local_args);
  assert(task->regions.size() == regions.size());
  float const *in_ptr = NULL;
  float *out_ptr = NULL, *gamma_ptr = NULL, *beta_ptr = NULL;
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  in_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  out_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  assert(in_domain == out_domain);
  assert(in_domain.get_volume() ==
         m->effective_num_elements * m->effective_batch_size);
  if (m->elementwise_affine) {
    assert(regions.size() == 4);
    Domain gamma_domain = runtime->get_index_space_domain(
        ctx, task->regions[2].region.get_index_space());
    gamma_ptr = helperGetTensorPointerRW<float>(
        regions[2], task->regions[2], FID_DATA, ctx, runtime);
    Domain beta_domain = runtime->get_index_space_domain(
        ctx, task->regions[3].region.get_index_space());
    beta_ptr = helperGetTensorPointerRW<float>(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
    assert(gamma_domain == beta_domain);
    assert(gamma_domain.get_volume() == m->effective_num_elements);
  } else {
    assert(regions.size() == 2);
  }

  forward_kernel_wrapper<float>(m, in_ptr, out_ptr, gamma_ptr, beta_ptr);
}

void LayerNorm::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(LAYERNORM_BWD_TASK_ID,
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
  // regions[1](I): input
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(1, FID_DATA);
  // regions[2](I/O): input_grad
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  if (elementwise_affine) {
    // regions[3](I): gamma
    launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[0]->region));
    launcher.add_field(3, FID_DATA);
    // regions[4](I/O): gamma_grad
    launcher.add_region_requirement(RegionRequirement(weights[0]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      weights[0]->region_grad));
    launcher.add_field(4, FID_DATA);
    // regions[5](I/O): beta_grad
    launcher.add_region_requirement(RegionRequirement(weights[1]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      weights[1]->region_grad));
    launcher.add_field(5, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): output_grad
  regions[1](I): input
  regions[2](I/O): input_grad
  regions[3](I): gamma
  regions[4](I/O): gamma_grad
  regions[5](I/O): beta_grad
   */
void LayerNorm::backward_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  LayerNormMeta const *m = *((LayerNormMeta **)task->local_args);
  assert(task->regions.size() == regions.size());
  float const *in_ptr = NULL, *out_grad_ptr = NULL, *gamma_ptr = NULL;
  float *in_grad_ptr = NULL, *gamma_grad_ptr = NULL, *beta_grad_ptr = NULL;
  Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  out_grad_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  in_ptr = helperGetTensorPointerRO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  Domain in_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  in_grad_ptr = helperGetTensorPointerRW<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  assert(in_domain == out_grad_domain);
  assert(in_domain.get_volume() ==
         m->effective_num_elements * m->effective_batch_size);
  if (m->elementwise_affine) {
    assert(regions.size() == 6);
    Domain gamma_domain = runtime->get_index_space_domain(
        ctx, task->regions[3].region.get_index_space());
    gamma_ptr = helperGetTensorPointerRO<float>(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
    Domain gamma_grad_domain = runtime->get_index_space_domain(
        ctx, task->regions[4].region.get_index_space());
    gamma_grad_ptr = helperGetTensorPointerRW<float>(
        regions[4], task->regions[4], FID_DATA, ctx, runtime);
    Domain beta_grad_domain = runtime->get_index_space_domain(
        ctx, task->regions[5].region.get_index_space());
    beta_grad_ptr = helperGetTensorPointerRW<float>(
        regions[5], task->regions[5], FID_DATA, ctx, runtime);
    assert(gamma_domain == gamma_grad_domain);
    assert(gamma_domain == beta_grad_domain);
    assert(gamma_domain.get_volume() == m->effective_num_elements);
  } else {
    assert(regions.size() == 3);
  }

  backward_kernel_wrapper<float>(m,
                                 out_grad_ptr,
                                 in_ptr,
                                 in_grad_ptr,
                                 gamma_ptr,
                                 gamma_grad_ptr,
                                 beta_grad_ptr);
}

bool LayerNorm::measure_operator_cost(Simulator *sim,
                                      MachineView const &mv,
                                      CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_input;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  LayerNormMeta *m = new LayerNormMeta(sim->handler, this);

  sim->free_all();
  float *in_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(in_ptr != NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *out_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(out_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  // FIXME please add gamma_ptr and beta_ptr after finish the implementation
  float *gamma_ptr = NULL, *beta_ptr = NULL;

  bool out_of_memory =
      (in_ptr == NULL) || (out_ptr == NULL) ||
      (((gamma_ptr == NULL) || (beta_ptr == NULL)) && (m->elementwise_affine));
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(m, in_ptr, out_ptr, gamma_ptr, beta_ptr);
  };

  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *in_grad_ptr =
        (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(in_grad_ptr != NULL);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *out_grad_ptr = NULL;
    out_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(out_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    float *gamma_grad_ptr = NULL, *beta_grad_ptr = NULL;

    out_of_memory = (in_grad_ptr == NULL) || (out_grad_ptr == NULL) ||
                    (((gamma_grad_ptr == NULL) || (beta_grad_ptr == NULL)) &&
                     (m->elementwise_affine));
    if (out_of_memory) {
      cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      return true;
    }

    backward = [&] {
      backward_kernel_wrapper<float>(m,
                                     out_grad_ptr,
                                     in_ptr,
                                     in_grad_ptr,
                                     gamma_ptr,
                                     gamma_grad_ptr,
                                     beta_grad_ptr);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure LayerNorm] name(%s) num_elements(%zu) "
                      "forward_time(%.4lf) backward_time(%.4lf)\n",
                      name,
                      sub_output.get_volume(),
                      cost_metrics.forward_time,
                      cost_metrics.backward_time);
  } else {
    log_measure.debug("[Measure LayerNorm] name(%s) num_elements(%zu) "
                      "forward_time(%.4lf)\n",
                      name,
                      sub_output.get_volume(),
                      cost_metrics.forward_time);
  }

  return true;
}

void LayerNorm::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->axes.size());
  for (size_t i = 0; i < this->axes.size(); i++) {
    sez.serialize(this->axes[i]);
  }
  sez.serialize(this->elementwise_affine);
  sez.serialize(this->eps);
}

using PCG::Node;
/*static*/
Node LayerNorm::deserialize(FFModel &ff,
                            Legion::Deserializer &dez,
                            ParallelTensor inputs[],
                            int num_inputs) {
  assert(num_inputs == 1);
  size_t num_axes;
  std::vector<int> axes;
  bool elementwise_affine;
  float eps;
  size_t id;
  dez.deserialize(id);
  LayerID layer_guid(id);
  dez.deserialize(num_axes);
  for (size_t i = 0; i < num_axes; i++) {
    int axis_idx;
    dez.deserialize(axis_idx);
    axes.push_back(axis_idx);
  }
  dez.deserialize(elementwise_affine);
  dez.deserialize(eps);

  LayerNormParams params;
  params.layer_guid = layer_guid;
  params.axes = axes;
  params.elementwise_affine = elementwise_affine;
  params.eps = eps;
  return ff.get_or_create_node<LayerNorm>(inputs[0], params);
}

Op *LayerNorm::materialize(FFModel &ff,
                           ParallelTensor inputs[],
                           int num_inputs) const {
  LayerNormParams params = get_params();
  return new LayerNorm(
      ff, params, inputs[0], this->name, true /*allocate_weights*/);
}

enum Slots {INPUT, OUTPUT, GAMMA, BETA, PER_DEVICE_STATE, ATTRS, HANDLE };

OpTaskInvocation init(LayerNormAttrs const & attrs) {
  OpTaskBinding b;

  b.bind_arg(HANDLE, ff_handle());
  b.bind_arg(ATTRS, attrs);

  return {LAYERNORM_INIT_TASK_ID, b};
}

OpTaskInvocation forward(LayerNormAttrs const & attrs) {
  OpTaskBinding b;

  b.bind(INPUT, input_tensor(0));
  b.bind(OUTPUT, output_tensor(0));
  b.bind(GAMMA, weight_tensor(0));//todo, this may have some problem
  b.bind(BETA, weight_tensor(1));//how to get gmmam and beta
  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(PER_DEVICE_STATE, per_device_state<LayerNormPerDeviceState>());

  return {LAYERNORM_FWD_TASK_ID, b};
}

OpTaskInvocation backward(LayerNormAttrs const & attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {LAYERNORM_BWD_TASK_ID, b};
}


static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
    auto input = acc.get_tensor<Permission::RO>(INPUT);
    auto output = acc.get_tensor<Permission::WO>(OUTPUT);
    auto gamma = acc.get_tensor<Permission::WO>(GAMMA);
    auto beta = acc.get_tensor<Permission::WO>(BETA);

    ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
    auto &state = acc.get_argument<LayerNormPerDeviceState>(PER_DEVICE_STATE);

    return profile(forward_kernel,
                  profiling,
                  "[LayerNorm] forward time = %.2lfms\n",
                  state,
                  input.get_float_ptr(),
                  output.get_float_ptr(),
                  gamma.get_float_ptr(),
                  beta.get_float_ptr());
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}


static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permission::RO>(INPUT);
  auto gamma = acc.get_tensor<Permission::RO>(GAMMA);

  auto input_grad = acc.get_tensor<Permission::RW>(INPUT_GRAD);
  auto gamma_grad = acc.get_tensor<Permission::RW>(GAMMA_GRAD);
  auto beta_grad = acc.get_tensor<Permission::RW>(BETA_GRAD);
  auto output_grad = acc.get_tensor<Permission::RO>(OUTPUT_GRAD);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto &state = acc.get_argument<LayerNormPerDeviceState>(PER_DEVICE_STATE);

  return profile(backward_kernel,
                  profiling,
                  "[LayerNorm] backward time = %.2lfms\n",
                  state,
                  output_grad.get_float_ptr(),
                  input.get_float_ptr(),
                  input_grad.get_float_ptr(),
                  gamma.get_float_ptr(),
                  gamma_grad.get_float_ptr(),
                  beta_grad.get_float_ptr());
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

static DeviceSpecific<LayerNormPerDeviceState> init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<MultiHeadAttentionAttrs>(ATTRS);
  Allocator allocator = acc.get_allocator();
  FFHandler handle = acc.get_argument<FFHandler>(HANDLE);
  //question: how to get batch_size and effective_num_elements
  int64_t effective_batch_size, effective_num_elements;

  DeviceSpecific<LayerNormPerDeviceState> per_device_state = 
      acc.create_device_specific<LayerNormPerDeviceState>(
        init_kernel(handle,
                    allocator,
                    attrs.elementwise_affine,
                    effective_batch_size,
                    effective_num_elements,
                    attrs.eps)
      );
}

static DeviceSpecific<LayerNormPerDeviceState>  init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  LayerNormAttrs const & attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
    auto env = sim.new_environment(); 
    ParallelTensorShape output_shape =get_output_shape(attrs, input_shape);

    SimTaskBinding init_binding;
    init_binding.bind_arg(HANDLE, ff_handle());
    init_binding.bind_arg(ATTRS, attrs);

    auto init_accessor = env.get_init_accessor(LAYERNORM_INIT_TASK_ID, init_binding);

    DeviceSpecific<LayerNormPerDeviceState> = init_task_impl(init_accessor);

    SimTaskBinding fwd_binding;
    fwd_binding.bind(INPUT, input_shape);
    fwd_binding.bind(OUTPUT, output_shape);
    //TODO how to handle gamma and beta, where are they from

    SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

    auto fwd_accessor = env.get_fwd_accessor(LAYERNORM_FWD_TASK_ID, fwd_binding);
    auto bwd_accessor = env.get_bwd_accessor(LAYERNORM_BWD_TASK_ID, bwd_binding);

    float forward_time = forward_task_impl(fwd_accessor).value();
    float backward_time = backward_task_impl(bwd_accessor).value();

    float sync_time = default_estimate_sync_time(env);
    return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<LAYERNORM_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);
  init.add_arg_slot<LayerNormAttrs>(ATTRS); 
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<LayerNormPerDeviceState>();

  register_task(LAYERNORM_INIT_TASK_ID, "LayerNorm init", init, init_task);
}

template <>
void register_task<LAYERNORM_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  //how to hande gamma and beta, this may have some problem
  fwd.add_input_slot(GAMMA);
  fwd.add_input_slot(BETA);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<LayerNormPerDeviceState>(PER_DEVICE_STATE);

  register_task(LAYERNORM_FWD_TASK_ID, "LayerNorm forward", fwd, forward_task);
}

template <>
void register_task<LAYERNORM_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(LAYERNORM_FWD_TASK_ID));

  register_task(LAYERNORM_BWD_TASK_ID, "LayerNorm backward", bwd, backward_task); 
}



}; // namespace FlexFlow
