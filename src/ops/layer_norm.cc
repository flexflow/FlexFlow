/* Copyright 2021 CMU, Facebook
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

#include "flexflow/ops/layer_norm.h"

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

Tensor FFModel::layer_norm(const Tensor input,
                           std::vector<int> const &axes,
                           bool elementwise_affine,
                           float eps,
                           char const *name) {
  // axes must be the last axes.size() dimensions
  for (int i = 0; i < axes.size(); i++) {
    bool found = false;
    for (int j = 0; j < axes.size(); j++)
      if (axes[j] == input->num_dims - 1 - i)
        found = true;
    if (!found) {
      assert(false && "axes must be the last axes.size() dimensions");
    }
  }
  int num_weights = elementwise_affine ? 2 : 0;
  Layer *ln = new Layer(this,
                        OP_LAYERNORM,
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
    for (int i = 0; i < axes.size(); i++)
      M *= input->dims[input->num_dims - 1 - axes[i]];
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
                     LayerID const &_layer_guid,
                     const ParallelTensor _input,
                     std::vector<int> const &axes,
                     bool _elementwise_affine,
                     float _eps,
                     bool allocate_weights,
                     char const *name)
    : Op(model,
         OP_LAYERNORM,
         name,
         1 /*inputs*/,
         _elementwise_affine ? 2 : 0 /*weights*/,
         1 /*outputs*/,
         _input),
      elementwise_affine(_elementwise_affine), eps(_eps) {
  // overwrite layer_guid
  layer_guid = _layer_guid;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, _input->dims, _input->data_type, this);
  assert(check_output_input_weight_parallel_dims(allocate_weights));
  ParallelDim output_dims[MAX_TENSOR_DIM];
  int M = 1;
  for (int i = 0; i < axes.size(); i++)
    M *= inputs[0]->dims[inputs[0]->num_dims - 1 - axes[i]].size;
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

#ifdef DEADCODE
void LayerNorm::create_weights(FFModel &model) {
  std::string pcname = name;
  task_is = model.get_or_create_task_is(outputs[0].numDim, pcname);

  // TODO: temp work, will let users to pick either NCCL or PS
#ifdef FF_USE_NCCL
  ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
  ParameterSyncType comm_type = ParameterSyncType::PS;
#endif

  // Create scale and bias
  Initializer *scale_initializer = new ConstantInitializer(1.0f);
  Initializer *bias_initializer = new ConstantInitializer(0.0f);
  int const dims[1] = {weights[0].adim[0]};
  switch (outputs[0].numDim) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    weights[0] = model.create_linear_weight<1, DIM>(this,                      \
                                                    dims,                      \
                                                    DT_FLOAT,                  \
                                                    scale_initializer,         \
                                                    true /*create_grad*/,      \
                                                    comm_type);                \
    weights[1] = model.create_linear_weight<1, DIM>(this,                      \
                                                    dims,                      \
                                                    DT_FLOAT,                  \
                                                    bias_initializer,          \
                                                    true /*create_grad*/,      \
                                                    comm_type);                \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
  }
}

void LayerNorm::create_output_and_partition(FFModel &model) {
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = model.get_or_create_task_is(outputs[0].numDim, pcname);
  Context ctx = model.config.lg_ctx;
  Runtime *runtime = model.config.lg_hlr;
  Domain part_rect = runtime->get_index_space_domain(ctx, task_is);
  {
    int dims[MAX_TENSOR_DIM];
    int ndims = outputs[0].numDim;
    for (int i = 0; i < outputs[0].numDim; i++)
      dims[i] = outputs[0].adim[ndims - 1 - i];
    switch (ndims) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    outputs[0] = model.create_tensor<DIM>(dims, outputs[0].data_type, this);   \
    break;                                                                     \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    }
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }
  Domain input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
  // Currently assume output and input must be partitioned in the same way
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0].part;
    input_grad_lps[0] = inputs[0].part_grad;
  } else {
    assert(false &&
           "LayerNorm currently assume output/input have same partition");
  }
}
#endif

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

OpMeta *LayerNorm::init_task(Task const *task,
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

  LayerNorm::forward_kernel_wrapper<float>(
      m, in_ptr, out_ptr, gamma_ptr, beta_ptr);
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

  LayerNorm::backward_kernel_wrapper<float>(m,
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
  return false;
}

}; // namespace FlexFlow
