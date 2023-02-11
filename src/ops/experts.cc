/* Copyright 2022 CMU
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

#include "flexflow/ops/experts.h"
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

// For now, we use one input and one output per expert
Tensor FFModel::experts(Tensor const *inputs,
                        int num_experts,
                        int experts_start_idx,
                        int experts_output_dim_size,
                        float alpha,
                        int experts_num_layers,
                        int experts_internal_dim_size,
                        char const *name) {

  // Check that there are three inputs: the input tensor, the indices and the
  // topk_gate_preds
  assert(inputs[0] != nullptr);
  int num_dims = inputs[0]->num_dims;
  assert(inputs[1]->num_dims == num_dims);
  assert(inputs[2]->num_dims == num_dims);
  int topk = inputs[1]->dims[0];
  assert(inputs[2]->dims[0] == topk);
  for (int i = 1; i < num_dims; i++) {
    assert(inputs[0]->dims[i] == inputs[1]->dims[i]);
    assert(inputs[1]->dims[i] == inputs[2]->dims[i]);
  }

  assert(inputs[1]->data_type == DT_INT32 || inputs[1]->data_type == DT_INT64);

  assert(experts_num_layers == 1 && "Multi-layer experts not implemented yet.");
  assert(experts_num_layers == 1 || experts_internal_dim_size > 0);

  Tensor fused_experts = this->dense(
      inputs[0], num_experts * experts_output_dim_size, AC_MODE_RELU);
  fused_experts = this->softmax(fused_experts);

  Tensor const layer_inputs[3] = {fused_experts, inputs[1], inputs[2]};

  Layer *e = new Layer(this,
                       OP_EXPERTS,
                       DT_FLOAT,
                       name,
                       3 /*inputs*/,
                       0 /*weights*/,
                       num_experts /*outputs*/,
                       layer_inputs);

  {
    int dims[MAX_TENSOR_DIM];
    for (int i = 1; i < num_dims; i++) {
      dims[i] = inputs[0]->dims[i];
    }
    dims[0] = experts_output_dim_size;
    for (int i = 0; i < num_experts; i++) {
      e->outputs[i] = create_tensor_legion_ordering(
          num_dims, dims, DT_FLOAT, e, 0, true /*create_grad*/);
      assert(e->outputs[i] != nullptr);
    }
  }

  e->add_int_property("num_experts", num_experts);
  e->add_int_property("experts_start_idx", experts_start_idx);
  e->add_int_property("experts_output_dim_size", experts_output_dim_size);
  e->add_float_property("alpha", alpha);
  e->add_int_property("experts_num_layers", experts_num_layers);
  e->add_int_property("experts_internal_dim_size", experts_internal_dim_size);
  layers.push_back(e);

  Tensor ret = e->outputs[0];
  for (int i = 1; i < num_experts; i++) {
    this->add(ret, e->outputs[i], /*inplace_a*/ true);
  }
  return ret;
}

Op *Experts::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("num_experts", value);
  int num_experts = value;
  layer->get_int_property("experts_start_idx", value);
  int experts_start_idx = value;
  layer->get_int_property("experts_output_dim_size", value);
  int experts_output_dim_size = value;
  float value2;
  layer->get_float_property("alpha", value2);
  float alpha = value2;
  layer->get_int_property("experts_num_layers", value);
  int experts_num_layers = value;
  layer->get_int_property("experts_internal_dim_size", value);
  int experts_internal_dim_size = value;
  return new Experts(model,
                     inputs.data(),
                     num_experts,
                     experts_start_idx,
                     experts_output_dim_size,
                     alpha,
                     experts_num_layers,
                     experts_internal_dim_size,
                     layer->name);
}

ExpertsParams Experts::get_params() const {
  ExpertsParams params;
  params.num_experts = num_experts;
  params.experts_start_idx = experts_start_idx;
  params.experts_output_dim_size = experts_output_dim_size;
  params.alpha = alpha;
  params.experts_num_layers = experts_num_layers;
  params.experts_internal_dim_size = experts_internal_dim_size;
  return params;
}

bool ExpertsParams::is_valid(
    std::vector<ParallelTensorShape> const &inputs) const {
  if (inputs.size() != 3) {
    printf("Number of inputs to the Experts layer is wrong\n");
    return false;
  }
  if (!inputs[0].is_valid()) {
    printf("The first tensor passed to the Experts layer is not valid\n");
    return false;
  }
  if (!inputs[1].is_valid()) {
    printf("The second tensor passed to the Experts layer is not valid\n");
    return false;
  }
  if (!inputs[2].is_valid()) {
    printf("The third tensor passed to the Experts layer is not valid\n");
    return false;
  }
  if (inputs[0].num_dims != inputs[1].num_dims ||
      inputs[1].num_dims != inputs[2].num_dims) {
    printf("Mismatch found between the number of dimensions of the three input "
           "tensors for the Expert layer\n");
    return false;
  }
  if (inputs[0].data_type != DT_FLOAT) {
    printf("Data type of the first input to the Experts layer is wrong!\n");
    return false;
  }
  if (inputs[1].data_type != DT_INT32 && inputs[1].data_type != DT_INT64) {
    printf("Data type of the second input to the Experts layer is wrong!\n");
    return false;
  }
  if (inputs[2].data_type != DT_FLOAT) {
    printf("Data type of the third input to the Experts layer is wrong!\n");
    return false;
  }
  if (inputs[0].dims[0].size != num_experts * experts_output_dim_size) {
    printf("Dimension 0 of input tensor 1 to the Experts layer is wrong.\n");
    return false;
  }
  if (inputs[1].dims[0] != inputs[2].dims[0]) {
    printf(
        "Dimension mismatch between indices and topk_gate_preds tensors passed "
        "to the Experts layer.\n");
    return false;
  }
  for (int i = 1; i < inputs[0].num_dims; i++) {
    if (inputs[0].dims[i] != inputs[1].dims[i] ||
        inputs[1].dims[i] != inputs[2].dims[i]) {
      printf("Dimension mismatch among the input tensors passed to the Experts "
             "layer.\n");
      return false;
    }
  }
  return true;
}

bool operator==(ExpertsParams const &lhs, ExpertsParams const &rhs) {
  return lhs.num_experts == rhs.num_experts &&
         lhs.experts_start_idx == rhs.experts_start_idx &&
         lhs.experts_output_dim_size == rhs.experts_output_dim_size &&
         lhs.alpha == rhs.alpha &&
         lhs.experts_num_layers == rhs.experts_num_layers &&
         lhs.experts_internal_dim_size == rhs.experts_internal_dim_size;
}

Experts::Experts(FFModel &model,
                 ExpertsParams const &params,
                 // std::pair<ParallelTensor, ParallelTensor> const &inputs,
                 std::vector<ParallelTensor> const &inputs,
                 char const *name)
    : Experts(model,
              inputs.data(),
              params.num_experts,
              params.experts_start_idx,
              params.experts_output_dim_size,
              params.alpha,
              params.experts_num_layers,
              params.experts_internal_dim_size,
              name) {}

Experts::Experts(FFModel &model,
                 ParallelTensor const *inputs,
                 int _num_experts,
                 int _experts_start_idx,
                 int _experts_output_dim_size,
                 float _alpha,
                 int _experts_num_layers,
                 int _experts_internal_dim_size,
                 char const *name)
    : Op(model,
         OP_EXPERTS,
         DT_FLOAT,
         name,
         3 /*inputs*/,
         0 /*weights*/,
         _num_experts /*outputs*/,
         inputs),
      num_experts(_num_experts), experts_start_idx(_experts_start_idx),
      experts_output_dim_size(_experts_output_dim_size), alpha(_alpha),
      experts_num_layers(_experts_num_layers),
      experts_internal_dim_size(_experts_internal_dim_size) {

  assert(num_experts > 0);
  assert(numInputs == 3);
  assert(numOutputs == num_experts);

  assert(inputs[0] != nullptr);
  int num_dims = inputs[0]->num_dims;
  assert(inputs[1]->num_dims == num_dims);
  assert(inputs[2]->num_dims == num_dims);

  int out_dim = num_experts * experts_output_dim_size;
  assert(inputs[0]->dims[0].size == out_dim);
  int topk = inputs[1]->dims[0].size;
  assert(inputs[2]->dims[0].size == topk);

  for (int i = 1; i < num_dims; i++) {
    assert(inputs[0]->dims[i] == inputs[1]->dims[i]);
    assert(inputs[1]->dims[i] == inputs[2]->dims[i]);
  }

  assert(inputs[1]->data_type == DT_INT32 || inputs[1]->data_type == DT_INT64);
  assert(experts_num_layers == 1 && "Multi-layer experts not implemented yet.");
  assert(experts_num_layers == 1 || experts_internal_dim_size > 0);

  // Assume that we don't parallelize the channel dim of input
  // nor the expert_assigned dim of indices
  assert(inputs[0]->dims[0].degree == 1);
  assert(inputs[1]->dims[0].degree == 1);
  assert(inputs[2]->dims[0].degree == 1);

  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < num_dims; i++) {
    dims[i] = inputs[0]->dims[i];
  }
  dims[0].size = experts_output_dim_size;
  // numOutputs = num_experts;
  // numWeights = 0;
  for (int i = 0; i < num_experts; i++) {
    outputs[i] = model.create_parallel_tensor_legion_ordering(
        num_dims, dims, inputs[0]->data_type, this, i /*owner_idx*/);
    assert(outputs[i] != nullptr);
  }
}

void Experts::serialize(Legion::Serializer &sez) const {
  ExpertsParams params = get_params();
  sez.serialize(params.num_experts);
  sez.serialize(params.experts_start_idx);
  sez.serialize(params.experts_output_dim_size);
  sez.serialize(params.alpha);
  sez.serialize(params.experts_num_layers);
  sez.serialize(params.experts_internal_dim_size);
}

using PCG::Node;
Node Experts::deserialize(FFModel &ff,
                          Legion::Deserializer &dez,
                          std::vector<ParallelTensor> const &inputs,
                          int num_inputs) {
  int num_experts, experts_start_idx, experts_output_dim_size,
      experts_num_layers, experts_internal_dim_size;
  float alpha;
  dez.deserialize(num_experts);
  dez.deserialize(experts_start_idx);
  dez.deserialize(experts_output_dim_size);
  dez.deserialize(alpha);
  dez.deserialize(experts_num_layers);
  dez.deserialize(experts_internal_dim_size);

  assert(num_inputs == 3);

  ExpertsParams params;
  params.num_experts = num_experts;
  params.experts_start_idx = experts_start_idx;
  params.experts_output_dim_size = experts_output_dim_size;
  params.alpha = alpha;
  params.experts_num_layers = experts_num_layers;
  params.experts_internal_dim_size = experts_internal_dim_size;

  return ff.get_or_create_node<Experts>(inputs, params);
}

void Experts::init_inference(FFModel const &ff,
                             std::vector<ParallelTensor> const &batch_inputs,
                             std::vector<ParallelTensor> const &batch_outputs,
                             MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = batch_outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  size_t machine_view_hash =
      mv ? mv->hash() : batch_outputs[0]->machine_view.hash();
  IndexLauncher launcher(EXPERTS_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Experts)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  // expert predictions
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // expert assignment indices
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // topk_gate_preds
  launcher.add_region_requirement(RegionRequirement(batch_inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  for (int i = 0; i < num_experts; i++) {
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[i]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[i]->region));
    launcher.add_field(i + 3, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Experts::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(EXPERTS_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Experts)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // expert predictions
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // expert assignment indices
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // topk_gate_preds
  launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  for (int i = 0; i < num_experts; i++) {
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[i]->region));
    launcher.add_field(i + 3, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *Experts::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  Experts const *exp = (Experts *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  ExpertsMeta *m = new ExpertsMeta(
      handle, exp->num_experts, exp->experts_start_idx, exp->alpha);
  m->profiling = exp->profiling;
  return m;
}

void Experts::forward(FFModel const &ff) {
  // assert(false && "Experts is designed for inference only");
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(EXPERTS_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // expert predictions
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // expert assignment indices
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // topk_gate_preds
  launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  for (int i = 0; i < num_experts; i++) {
    // expert output per token (only the chosen experts have non-zero
    // contributions)
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[i]->region));
    launcher.add_field(i + 3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Experts::inference(FFModel const &ff,
                        std::vector<ParallelTensor> const &batch_inputs,
                        std::vector<ParallelTensor> const &batch_outputs,
                        MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  size_t machine_view_hash =
      mv ? mv->hash() : batch_outputs[0]->machine_view.hash();
  std::cout << "Experts op machine_view: " << *(MachineView const *)mv
            << std::endl;
  // std::cout << "machine_view hash passed: " << mv->hash() << " machine view
  // gotten: " << machine_view_hash
  //           << std::endl;
  IndexLauncher launcher(EXPERTS_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  // expert predictions
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // expert assignment indices
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // topk_gate_preds
  launcher.add_region_requirement(RegionRequirement(batch_inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  for (int i = 0; i < num_experts; i++) {
    // expert output per token (only the chosen experts have non-zero
    // contributions)
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[i]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[i]->region));
    launcher.add_field(i + 3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Experts::inference_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  assert(regions.size() == task->regions.size());
  int num_experts = regions.size() - 3;

  ExpertsMeta const *m = *((ExpertsMeta **)task->local_args);

  // get input, indices, topk_gate_preds
  float const *input_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  int const *indices_ptr = helperGetTensorPointerRO<int>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  float const *topk_gate_pred_ptr = helperGetTensorPointerRO<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);

  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain indices_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain topk_gate_pred_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  int input_dims = input_domain.get_dim();
  int indices_dims = indices_domain.get_dim();
  int topk_gate_pred_dims = topk_gate_pred_domain.get_dim();
  assert(input_dims == indices_dims);
  assert(indices_dims == topk_gate_pred_dims);

  int replica_dim = input_dims - 1;
  int samples_index = input_dims - 2;

  coord_t out_dim =
      (input_domain.hi()[0] - input_domain.lo()[0] + 1) / num_experts;
  coord_t batch_size =
      input_domain.hi()[samples_index] - input_domain.lo()[samples_index] + 1;
  coord_t chosen_experts = indices_domain.hi()[0] - indices_domain.lo()[0];
  assert(chosen_experts ==
         topk_gate_pred_domain.hi()[0] - topk_gate_pred_domain.lo()[0]);

  for (int i = 1; i < input_dims; i++) {
    int a = input_domain.hi()[i] - input_domain.lo()[i] + 1;
    int b = indices_domain.hi()[i] - indices_domain.lo()[i] + 1;
    int c = topk_gate_pred_domain.hi()[i] - topk_gate_pred_domain.lo()[i] + 1;
    assert(a == b && b == c);
    if (i >= 1 && i < samples_index) {
      batch_size *= a;
    }
  }

  assert(batch_size <= MAX_BATCH_SIZE &&
         "batch size exceeds MAX_BATCH_SIZE defined in experts.h");
  assert(
      num_experts <= MAX_EXPERTS_PER_BLOCK &&
      "number of experts exceeds MAX_EXPERTS_PER_BLOCK defined in experts.h");

  float *outputs[num_experts];
  for (int i = 0; i < num_experts; i++) {
    Domain output_domain = runtime->get_index_space_domain(
        ctx, task->regions[3 + i].region.get_index_space());
    assert((output_domain.hi()[0] - output_domain.lo()[0] + 1) == out_dim);
    for (int j = 1; j < input_dims; j++) {
      int a = input_domain.hi()[j] - input_domain.lo()[j] + 1;
      int b = output_domain.hi()[j] - output_domain.lo()[j] + 1;
      assert(a == b);
    }
    outputs[i] = helperGetTensorPointerWO<float>(
        regions[3 + i], task->regions[3 + i], FID_DATA, ctx, runtime);
    assert(outputs[i] != nullptr);
  }
  return;
  Experts::forward_kernel_wrapper(m,
                                  input_ptr,
                                  indices_ptr,
                                  topk_gate_pred_ptr,
                                  outputs,
                                  chosen_experts,
                                  batch_size,
                                  out_dim);
}

void Experts::forward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(false && "Experts is designed for inference only");
}

void Experts::backward(FFModel const &ff) {
  assert(false && "Experts is designed for inference only");
}

void Experts::backward_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(false && "Experts is designed for inference only");
}

void Experts::print_layer(FFModel const &ff) {
  return;
}

bool Experts::measure_operator_cost(Simulator *sim,
                                    MachineView const &c,
                                    CostMetrics &cost_metrics) const {
  // This is an inference only operator
  assert(false && "Experts is designed for inference only");
  return false;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ExpertsParams>::operator()(
    FlexFlow::ExpertsParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.num_experts);
  hash_combine(key, params.experts_start_idx);
  hash_combine(key, params.experts_output_dim_size);
  hash_combine(key, params.alpha);
  hash_combine(key, params.experts_num_layers);
  hash_combine(key, params.experts_internal_dim_size);
  return key;
}
}; // namespace std
