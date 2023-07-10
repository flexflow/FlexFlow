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

#include "fused.h"
#include "kernels/accessor.h"
#include "kernels/batch_matmul_kernels.h"
#include "kernels/batch_norm_kernels.h"
#include "kernels/concat_kernels.h"
#include "kernels/conv_2d_kernels.h"
#include "kernels/cuda_helper.h"
#include "kernels/dropout_kernels.h"
#include "kernels/element_binary_kernels.h"
#include "kernels/element_unary_kernels.h"
#include "kernels/embedding_kernels.h"
#include "kernels/flat_kernels.h"
#include "kernels/linear_kernels.h"
#include "kernels/pool_2d_kernels.h"
#include "kernels/reshape_kernels.h"
#include "kernels/transpose_kernels.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::LogicalPartition;
using Legion::LogicalRegion;
using Legion::PhysicalRegion;
using Legion::PointInRectIterator;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

FusedOp::FusedOp(FFModel &model, Op *op)
    : Op(model,
         OP_FUSED,
         DT_NONE,
         op->name,
         0 /*weights*/,
         0 /*weights*/,
         0 /*outputs*/) {
  numInputs = op->numInputs;
  for (int i = 0; i < numInputs; i++) {
    inputs[i] = op->inputs[i];
    input_data_types[i] = op->inputs[i]->data_type;
    // input_lps[i] = op->input_lps[i];
    // input_grad_lps[i] = op->input_grad_lps[i];
  }
  numWeights = op->numWeights;
  for (int i = 0; i < numWeights; i++) {
    weights[i] = op->weights[i];
    // weights[i]->owner_op = this;
    // weights[i]->owner_idx = i;
    weight_data_types[i] = op->weights[i]->data_type;
  }
  numOutputs = op->numOutputs;
  for (int i = 0; i < numOutputs; i++) {
    outputs[i] = op->outputs[i];
    outputs[i]->owner_op = this;
    outputs[i]->owner_idx = i;
    output_data_types[i] = op->outputs[i]->data_type;
  }
  numOperators = 1;
  op_num_inputs[0] = numInputs;
  op_num_weights[0] = numWeights;
  op_num_outputs[0] = numOutputs;
  op_op_type[0] = op->op_type;
  operators[0] = op;
  for (int i = 0; i < numInputs; i++) {
    op_input_source[i] = SOURCE_INPUT;
    op_input_idx[i] = i;
  }
  for (int i = 0; i < numWeights; i++) {
    op_weight_source[i] = SOURCE_WEIGHT;
    op_weight_idx[i] = i;
  }
  for (int i = 0; i < numOutputs; i++) {
    op_output_source[i] = SOURCE_OUTPUT;
    op_output_idx[i] = i;
  }
}

bool FusedOp::add_operator(FFModel &model, Op *op) {
  // Context ctx = model.config.lg_ctx;
  // Runtime* runtime = model.config.lg_hlr;
  //  Currently assume fusion optimization is performed
  //  after map_tensors
  //  So parallel_is and op->parallel_is are not empty
  // Domain my_domain = runtime->get_index_space_domain(ctx,
  // outputs[0]->parallel_is); Domain op_domain =
  // runtime->get_index_space_domain(ctx, op->outputs[0]->parallel_is);
  // ParallelConfig my_config, op_config;
  // assert(model.config.find_parallel_config(my_domain.get_dim(), name,
  // my_config)); assert(model.config.find_parallel_config(op_domain.get_dim(),
  // op->name, op_config));
  // Cannot fuse parallel operators since they have different paralel_is
  // in forward and backward
  assert(!op->is_parallel_op());
  // Currently don't consider nested fusion
  assert(op->op_type != OP_FUSED);
  MachineView my_view = outputs[0]->machine_view;
  MachineView op_view = op->outputs[0]->machine_view;
  if (my_view == op_view) {
    // Do nothing
  } else {
    return false;
  }
  int input_offset = 0, weight_offset = 0, output_offset = 0;
  for (int i = 0; i < numOperators; i++) {
    input_offset += op_num_inputs[i];
    weight_offset += op_num_weights[i];
    output_offset += op_num_outputs[i];
  }
  if ((input_offset + op->numInputs > MAX_NUM_FUSED_TENSORS) ||
      (weight_offset + op->numWeights > MAX_NUM_FUSED_TENSORS) ||
      (output_offset + op->numOutputs > MAX_NUM_FUSED_TENSORS)) {
    fprintf(stderr, "Cannot fuse. Consider increase MAX_NUM_FUSED_TENSORS\n");
    return false;
  }
  if (numOperators + 1 > MAX_NUM_FUSED_OPERATORS) {
    fprintf(
        stderr,
        "Reach to the fusion limit. Consider increase MAX_NUM_FUSED_OPERATORS");
    return false;
  }
  // Set inputs
  for (int i = 0; i < op->numInputs; i++) {
    bool found = false;
    for (int j = 0; j < numInputs; j++) {
      if (inputs[j]->region == op->inputs[i]->region) {
        // This input is one of my inputs
        assert(!found);
        assert(inputs[j]->region != LogicalRegion::NO_REGION);
        op_input_source[input_offset + i] = SOURCE_INPUT;
        op_input_idx[input_offset + i] = j;
        found = true;
        break;
      }
    }
    for (int j = 0; j < numOutputs; j++) {
      if ((outputs[j]->region == op->inputs[i]->region) && (!found)) {
        // This input is one of my outputs
        assert(!found);
        assert(outputs[j]->region != LogicalRegion::NO_REGION);
        op_input_source[input_offset + i] = SOURCE_OUTPUT;
        op_input_idx[input_offset + i] = j;
        found = true;
        break;
      }
    }
    if (found) {
      // Do nothing
    } else {
      inputs[numInputs] = op->inputs[i];
      input_data_types[numInputs] = op->inputs[i]->data_type;
      // input_lps[numInputs] = op->input_lps[i];
      // input_grad_lps[numInputs] = op->input_grad_lps[i];
      op_input_source[input_offset + i] = SOURCE_INPUT;
      op_input_idx[input_offset + i] = numInputs;
      numInputs += 1;
    }
  }
  // Set weights
  for (int i = 0; i < op->numWeights; i++) {
    bool found = false;
    for (int j = 0; j < numWeights; j++) {
      if (weights[j]->region == op->weights[i]->region) {
        assert(!found);
        assert(weights[j]->region != LogicalRegion::NO_REGION);
        op_weight_source[weight_offset + i] = SOURCE_WEIGHT;
        op_weight_idx[weight_offset + i] = j;
        found = true;
        break;
      }
    }
    if (found) {
      // Do nothing
    } else {
      weights[numWeights] = op->weights[i];
      // weights[numWeights]->owner_op = this;
      // weights[numWeights]->owner_idx = numWeights;
      weight_data_types[numWeights] = op->weights[i]->data_type;
      op_weight_source[weight_offset + i] = SOURCE_WEIGHT;
      op_weight_idx[weight_offset + i] = numWeights;
      numWeights += 1;
    }
  }
  // Set outputs
  for (int i = 0; i < op->numOutputs; i++) {
    bool found = false;
    for (int j = 0; j < numOutputs; j++) {
      if (outputs[j]->region == op->outputs[i]->region) {
        assert(!found);
        found = true;
        op_output_source[output_offset + i] = SOURCE_OUTPUT;
        op_output_idx[output_offset + i] = j;
      }
    }
    if (found) {
      continue;
    }
    outputs[numOutputs] = op->outputs[i];
    outputs[numOutputs]->owner_op = this;
    outputs[numOutputs]->owner_idx = numOutputs;
    output_data_types[numOutputs] = op->outputs[i]->data_type;
    op_output_source[output_offset + i] = SOURCE_OUTPUT;
    op_output_idx[output_offset + i] = numOutputs;
    numOutputs += 1;
  }
  assert(op->numInputs > 0);
  assert(op->numWeights >= 0);
  assert(op->numOutputs > 0);
  op_num_inputs[numOperators] = op->numInputs;
  op_num_weights[numOperators] = op->numWeights;
  op_num_outputs[numOperators] = op->numOutputs;
  op_op_type[numOperators] = op->op_type;
  operators[numOperators] = op;
  numOperators += 1;
  assert(numOperators <= MAX_NUM_FUSED_OPERATORS);
  if (numInputs > MAX_NUM_INPUTS) {
    fprintf(stderr,
            "Reach to the #inputs limit during fusion.\n"
            "Consider increase MAX_NUM_INPUTS to allow more fusions.\n");
    return false;
  }
  if (numWeights > MAX_NUM_WEIGHTS) {
    fprintf(stderr,
            "Reach to the #weights limit during fusion.\n"
            "Consider increase MAX_NUM_WEIGHTS to allow more fusions.\n");
    return false;
  }
  if (numOutputs > MAX_NUM_OUTPUTS) {
    fprintf(stderr,
            "Reach to the #outputs limit during fusion.\n"
            "Consider increase MAX_NUM_OUTPUTS to allow more fusions.\n");
  }
  return true;
}

void FusedOp::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Call init methods in individual operators
  Domain domain = runtime->get_index_space_domain(ctx, parallel_is);
  for (int i = 0; i < numOperators; i++) {
    operators[i]->init(ff);
    for (size_t j = 0; j < domain.get_volume(); j++) {
      fused_meta[j].meta[i] = operators[i]->meta[j];
    }
  }
  for (size_t j = 0; j < domain.get_volume(); j++) {
    fused_meta[j].numOperators = numOperators;
  }
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    int idx = 0;                                                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      argmap.set_point(*it,                                                    \
                       TaskArgument(&fused_meta[idx++], sizeof(FusedOpMeta))); \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(FUSEDOP_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(FusedOp)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    int idx = 0;                                                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      meta[idx++] = fm.get_result<PerDeviceOpState *>(*it);                    \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

void FusedOp::forward(FFModel const &ff) {
  // Set iter_config
  iter_config = ff.iter_config;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(FUSEDOP_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  int offset = 0;
  for (int i = 0; i < numInputs; i++) {
    assert(inputs[i]->part != LogicalPartition::NO_PART);
    assert(inputs[i]->region != LogicalRegion::NO_REGION);
    launcher.add_region_requirement(RegionRequirement(inputs[i]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      inputs[i]->region));
    launcher.add_field(offset + i, FID_DATA);
  }
  offset += numInputs;
  for (int i = 0; i < numWeights; i++) {
    assert(weights[i]->region != LogicalRegion::NO_REGION);
    launcher.add_region_requirement(RegionRequirement(weights[i]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[i]->region));
    launcher.add_field(offset + i, FID_DATA);
  }
  offset += numWeights;
  for (int i = 0; i < numOutputs; i++) {
    assert(outputs[i]->region != LogicalRegion::NO_REGION);
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[i]->region));
    launcher.add_field(offset + i, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void FusedOp::backward(FFModel const &ff) {
  // Set iter_config
  iter_config = ff.iter_config;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(FUSEDOP_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(FusedOp)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  int idx = 0;
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(RegionRequirement(inputs[i]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      inputs[i]->region));
    launcher.add_field(idx++, FID_DATA);
  }
  for (int i = 0; i < numWeights; i++) {
    launcher.add_region_requirement(RegionRequirement(weights[i]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[i]->region));
    launcher.add_field(idx++, FID_DATA);
  }
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[i]->region));
    launcher.add_field(idx++, FID_DATA);
  }
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(RegionRequirement(inputs[i]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      inputs[i]->region_grad));
    launcher.add_field(idx++, FID_DATA);
  }
  for (int i = 0; i < numWeights; i++) {
    launcher.add_region_requirement(RegionRequirement(weights[i]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      weights[i]->region_grad));
    launcher.add_field(idx++, FID_DATA);
  }
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      outputs[i]->region_grad));
    launcher.add_field(idx++, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

bool FusedOp::measure_operator_cost(Simulator *sim,
                                    MachineView const &mv,
                                    CostMetrics &cost_metrics) const {
  // The search should happen before fusion
  assert(false);
  return false;
}

PerDeviceOpState *FusedOp::init_task(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  FusedOp const *fused = (FusedOp *)task->args;
  FusedOpPerDeviceState const *metas =
      (FusedOpPerDeviceState *)task->local_args;
  FusedOpPerDeviceState *local_meta = new FusedOpPerDeviceState();
  memcpy(local_meta, metas, sizeof(FusedOpPerDeviceState));
  local_meta->fused_op = (FusedOp *)malloc(sizeof(FusedOp));
  memcpy(static_cast<void *>(local_meta->fused_op),
         static_cast<void const *>(fused),
         sizeof(FusedOp));
  return ((PerDeviceOpState *)local_meta);
}

void FusedOp::forward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  // const FusedOp* fused = (FusedOp*) task->args;
  FusedOpPerDeviceState const *metas =
      *((FusedOpPerDeviceState **)task->local_args);
  FusedOp const *fused = metas->fused_op;
  assert(metas->numOperators == fused->numOperators);
  assert(regions.size() == task->regions.size());
  assert((int)regions.size() ==
         fused->numInputs + fused->numWeights + fused->numOutputs);
  // Domain input_domain[MAX_NUM_INPUTS];
  // Domain weight_domain[MAX_NUM_WEIGHTS];
  // Domain output_domain[MAX_NUM_OUTPUTS];
  GenericTensorAccessorR input_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorR weight_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorW output_accessor[MAX_NUM_OUTPUTS];
  assert(fused->numInputs <= MAX_NUM_INPUTS);
  for (int i = 0; i < fused->numInputs; i++) {
    // input_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i].region.get_index_space());
    input_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->input_data_types[i],
                                         regions[i],
                                         task->regions[i],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  int roff = fused->numInputs;
  assert(fused->numWeights <= MAX_NUM_WEIGHTS);
  for (int i = 0; i < fused->numWeights; i++) {
    // weight_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i + roff].region.get_index_space());
    weight_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->weight_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  roff += fused->numWeights;
  assert(fused->numOutputs <= MAX_NUM_OUTPUTS);
  for (int i = 0; i < fused->numOutputs; i++) {
    // output_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i + roff].region.get_index_space());
    output_accessor[i] =
        helperGetGenericTensorAccessorWO(fused->output_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  // Assert that all meta share the same dnn/blas handler
  int start = 0;
  for (start = 0; start < fused->numOperators; start++) {
    if (metas->meta[start] != NULL) {
      break;
    }
  }
  for (int op = start + 1; op < fused->numOperators; op++) {
    if (metas->meta[op] != NULL) {
      assert(metas->meta[start]->handle.blas == metas->meta[op]->handle.blas);
      assert(metas->meta[start]->handle.dnn == metas->meta[op]->handle.dnn);
    }
  }

  int ioff = 0, woff = 0, ooff = 0;
  for (int op = 0; op < fused->numOperators; op++) {
    // Domain my_id[MAX_NUM_INPUTS];
    // Domain my_wd[MAX_NUM_WEIGHTS];
    // Domain my_od[MAX_NUM_OUTPUTS];
    GenericTensorAccessorR my_input_accessor[MAX_NUM_INPUTS];
    GenericTensorAccessorR my_weight_accessor[MAX_NUM_WEIGHTS];
    GenericTensorAccessorW my_output_accessor[MAX_NUM_OUTPUTS];
    for (int i = 0; i < fused->op_num_inputs[op]; i++) {
      int my_off = fused->op_input_idx[i + ioff];
      if (fused->op_input_source[i + ioff] == SOURCE_INPUT) {
        // my_id[i] = input_domain[my_off];
        my_input_accessor[i] = input_accessor[my_off];
      } else if (fused->op_input_source[i + ioff] == SOURCE_OUTPUT) {
        // my_id[i] = output_domain[my_off];
        my_input_accessor[i] = output_accessor[my_off];
      } else {
        assert(false);
      }
    }
    for (int i = 0; i < fused->op_num_weights[op]; i++) {
      assert(fused->op_weight_source[i + woff] == SOURCE_WEIGHT);
      // my_wd[i] = weight_domain[fused->op_weight_idx[i + woff]];
      // my_wp[i] = weight_ptr[fused->op_weight_idx[i + woff]];
      my_weight_accessor[i] = weight_accessor[fused->op_weight_idx[i + woff]];
    }
    for (int i = 0; i < fused->op_num_outputs[op]; i++) {
      assert(fused->op_output_source[i + ooff] == SOURCE_OUTPUT);
      // my_od[i] = output_domain[fused->op_output_idx[i + ooff]];
      // my_op[i] = output_ptr[fused->op_output_idx[i + ooff]];
      my_output_accessor[i] = output_accessor[i + ooff];
    }
    switch (fused->op_op_type[op]) {
      case OP_CONCAT: {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        ConcatPerDeviceState *m = (ConcatPerDeviceState *)metas->meta[op];
        int num_inputs = fused->op_num_inputs[op];
        Kernels::Concat::forward_kernel(m,
                                        my_output_accessor[0],
                                        my_input_accessor,
                                        num_inputs,
                                        m->legion_axis);
        break;
      }
      case OP_CONV2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_dim() == 5);
        assert(my_weight_accessor[0].domain.get_dim() == 5);
        assert(my_output_accessor[0].domain.get_dim() == 5);
        Conv2DPerDeviceState *m = (Conv2DPerDeviceState *)metas->meta[op];
        Kernels::Conv2D::forward_kernel(m,
                                        my_input_accessor[0].get_float_ptr(),
                                        my_output_accessor[0].get_float_ptr(),
                                        my_weight_accessor[0].get_float_ptr(),
                                        my_weight_accessor[1].get_float_ptr());
        break;
      }
      case OP_BATCHNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_dim() == 5);
        assert(my_output_accessor[0].domain.get_dim() == 5);
        assert(my_weight_accessor[0].domain.get_dim() == 2);
        assert(my_weight_accessor[1].domain.get_dim() == 2);
        BatchNormPerDeviceState *m = (BatchNormPerDeviceState *)metas->meta[op];
        Kernels::BatchNorm::Internal::forward_kernel(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_weight_accessor[0].get_float_ptr(),
            my_weight_accessor[1].get_float_ptr());
        break;
      }
      case OP_DROPOUT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        Dropout *m = (DropoutPerDeviceState *)metas->meta[op];
        Kernels::Dropout::forward_kernel(m,
                                         my_input_accessor[0].get_float_ptr(),
                                         my_output_accessor[0].get_float_ptr());
        break;
      }
      case OP_LINEAR: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        Domain kernel_domain = my_weight_accessor[0].domain;
        int in_dim = kernel_domain.hi()[0] - kernel_domain.lo()[0] + 1;
        int out_dim = kernel_domain.hi()[1] - kernel_domain.lo()[1] + 1;
        int batch_size = my_input_accessor[0].domain.get_volume() / in_dim;
        assert(my_output_accessor[0].domain.get_volume() ==
               out_dim * batch_size);
        assert(my_input_accessor[0].domain.get_volume() == in_dim * batch_size);
        float const *bias_ptr = nullptr;
        if (fused->op_num_weights[op] == 2) {
          assert(my_weight_accessor[1].domain.get_volume() == out_dim);
          bias_ptr = my_weight_accessor[1].get_float_ptr();
        } else {
          assert(fused->op_num_weights[op] == 1);
        }
        LinearPerDeviceState *m = (LinearPerDeviceState *)metas->meta[op];
        Kernels::Linear::forward_kernel(m,
                                        my_input_accessor[0].get_float_ptr(),
                                        my_output_accessor[0].get_float_ptr(),
                                        my_weight_accessor[0].get_float_ptr(),
                                        bias_ptr,
                                        in_dim,
                                        out_dim,
                                        batch_size);
        break;
      }
      case OP_BATCHMATMUL: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        Domain out_domain = my_output_accessor[0].domain;
        Domain a_domain = my_input_accessor[0].domain;
        Domain b_domain = my_input_accessor[1].domain;
        int m = b_domain.hi()[0] - b_domain.lo()[0] + 1;
        assert(m == out_domain.hi()[0] - out_domain.lo()[0] + 1);
        int n = a_domain.hi()[1] - a_domain.lo()[1] + 1;
        assert(n == out_domain.hi()[1] - out_domain.lo()[1] + 1);
        int k = a_domain.hi()[0] - a_domain.lo()[0] + 1;
        assert(k == b_domain.hi()[1] - b_domain.lo()[1] + 1);
        assert(a_domain.get_dim() == b_domain.get_dim());
        assert(a_domain.get_dim() == out_domain.get_dim());
        int batch = 1;
        for (int i = 2; i < a_domain.get_dim(); i++) {
          int dim_size = a_domain.hi()[i] - a_domain.lo()[i] + 1;
          assert(dim_size == b_domain.hi()[i] - b_domain.lo()[i] + 1);
          assert(dim_size == out_domain.hi()[i] - out_domain.lo()[i] + 1);
          batch *= dim_size;
        }
        BatchMatmulPerDeviceState *meta =
            (BatchMatmulPerDeviceState *)metas->meta[op];
        Kernels::BatchMatmul::forward_kernel(
            meta,
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].get_float_ptr(),
            my_input_accessor[1].get_float_ptr(),
            (float const *)nullptr,
            m,
            n,
            k,
            batch,
            meta->a_seq_length_dim,
            meta->b_seq_length_dim,
            fused->iter_config.seq_length);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_DIV:
      case OP_EW_MAX:
      case OP_EW_MIN: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain == my_input_accessor[1].domain);
        assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        ElementBinaryPerDeviceState *m =
            (ElementBinaryPerDeviceState *)metas->meta[op];
        Kernels::ElementBinary::forward_kernel(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_input_accessor[1].get_float_ptr(),
            my_output_accessor[0].get_float_ptr());
        break;
      }
      case OP_EMBEDDING: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        EmbeddingPerDeviceState *m = (EmbeddingPerDeviceState *)metas->meta[op];
        if (m->aggr == AGGR_MODE_NONE) {
          // assert(kernel_domain.get_dim() == 2);
          assert(my_input_accessor[0].domain.get_dim() + 1 ==
                 my_output_accessor[0].domain.get_dim());
          for (size_t i = 0; i < my_input_accessor[0].domain.get_dim(); i++) {
            assert(my_input_accessor[0].domain.hi()[i] ==
                   my_output_accessor[0].domain.hi()[i + 1]);
            assert(my_input_accessor[0].domain.lo()[i] ==
                   my_output_accessor[0].domain.lo()[i + 1]);
          }
          assert(my_weight_accessor[0].domain.hi()[0] -
                     my_weight_accessor[0].domain.lo()[0] ==
                 my_output_accessor[0].domain.hi()[0] -
                     my_output_accessor[0].domain.lo()[0]);
        } else {
          assert(my_input_accessor[0].domain.get_dim() ==
                 my_output_accessor[0].domain.get_dim());
          for (size_t i = 1; i < my_input_accessor[0].domain.get_dim(); i++) {
            assert(my_input_accessor[0].domain.hi()[i] ==
                   my_output_accessor[0].domain.hi()[i]);
            assert(my_input_accessor[0].domain.lo()[i] ==
                   my_output_accessor[0].domain.lo()[i]);
          }
          assert(my_weight_accessor[0].domain.hi()[0] -
                     my_weight_accessor[0].domain.lo()[0] ==
                 my_output_accessor[0].domain.hi()[0] -
                     my_output_accessor[0].domain.lo()[0]);
        }
        int in_dim, out_dim, effective_batch_size;
        if (m->aggr == AGGR_MODE_NONE) {
          in_dim = 1;
          out_dim = my_output_accessor[0].domain.hi()[0] -
                    my_output_accessor[0].domain.lo()[0] + 1;
          effective_batch_size =
              my_output_accessor[0].domain.get_volume() / out_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].domain.get_volume());
        } else {
          assert(m->aggr == AGGR_MODE_AVG || m->aggr == AGGR_MODE_SUM);
          in_dim = my_input_accessor[0].domain.hi()[0] -
                   my_input_accessor[0].domain.lo()[0] + 1;
          out_dim = my_output_accessor[0].domain.hi()[0] -
                    my_output_accessor[0].domain.lo()[0] + 1;
          effective_batch_size =
              my_output_accessor[0].domain.get_volume() / out_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].domain.get_volume());
        }

        assert(my_input_accessor[0].data_type == DT_INT64);
        Kernels::Embedding::forward_kernel(m,
                                           my_input_accessor[0],
                                           my_output_accessor[0],
                                           my_weight_accessor[0],
                                           in_dim,
                                           out_dim,
                                           effective_batch_size);
        break;
      }
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_ELU: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        ElementUnaryPerDeviceState *m =
            (ElementUnaryPerDeviceState *)metas->meta[op];
        Kernels::ElementUnary::forward_kernel(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].domain.get_volume());
        break;
      }
      case OP_POOL2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        // assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        Pool2DPerDeviceState *m = (Pool2DPerDeviceState *)metas->meta[op];
        Kernels::Pool2D::forward_kernel(m,
                                        my_input_accessor[0].get_float_ptr(),
                                        my_output_accessor[0].get_float_ptr());
        break;
      }
      case OP_FLAT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_volume() ==
               my_output_accessor[0].domain.get_volume());
        Kernels::Flat::forward_kernel(my_input_accessor[0].get_float_ptr(),
                                      my_output_accessor[0].get_float_ptr(),
                                      my_input_accessor[0].domain.get_volume());
        break;
      }
      case OP_RESHAPE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_volume() ==
               my_output_accessor[0].domain.get_volume());
        Kernels::Reshape::forward_kernel(
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].domain.get_volume());
        break;
      }
      case OP_TRANSPOSE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_volume() ==
               my_output_accessor[0].domain.get_volume());
        TransposePerDeviceState *m = (TransposePerDeviceState *)metas->meta[op];
        Kernels::Transpose::forward_kernel(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].domain,
            my_output_accessor[0].domain);
        break;
      }
      default: {
        fprintf(stderr,
                "Fusion currently does not support type = %d\n",
                fused->op_op_type[op]);
        assert(false && "Fusion currently does not support type");
      }
    }
    ioff += fused->op_num_inputs[op];
    woff += fused->op_num_weights[op];
    ooff += fused->op_num_outputs[op];
  }
  // for (int i = 0; i < fused->numOutputs; i++)
  //   print_tensor<float>(output_ptr[i], output_domain[i].get_volume(),
  //   "[Fused:forward:output]");
}

void FusedOp::backward_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  // const FusedOp* fused = (FusedOp*) task->args;
  FusedOpPerDeviceState const *metas =
      *((FusedOpPerDeviceState **)task->local_args);
  FusedOp const *fused = metas->fused_op;

  assert(metas->numOperators == fused->numOperators);
  assert(regions.size() == task->regions.size());
  {
    int sum = fused->numInputs + fused->numWeights + fused->numOutputs;
    assert(sum * 2 == (int)regions.size());
  }
  // Domain input_domain[MAX_NUM_INPUTS], input_grad_domain[MAX_NUM_INPUTS];
  // Domain weight_domain[MAX_NUM_WEIGHTS], weight_grad_domain[MAX_NUM_WEIGHTS];
  // Domain output_domain[MAX_NUM_OUTPUTS], output_grad_domain[MAX_NUM_OUTPUTS];
  GenericTensorAccessorR input_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorW input_grad_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorR weight_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorW weight_grad_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorR output_accessor[MAX_NUM_OUTPUTS];
  GenericTensorAccessorW output_grad_accessor[MAX_NUM_OUTPUTS];
  int roff = 0;
  assert(fused->numInputs <= MAX_NUM_INPUTS);
  for (int i = 0; i < fused->numInputs; i++) {
    // input_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i].region.get_index_space());
    input_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->input_data_types[i],
                                         regions[i],
                                         task->regions[i],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  roff += fused->numInputs;
  assert(fused->numWeights <= MAX_NUM_WEIGHTS);
  for (int i = 0; i < fused->numWeights; i++) {
    // weight_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i + roff].region.get_index_space());
    weight_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->weight_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  roff += fused->numWeights;
  assert(fused->numOutputs <= MAX_NUM_OUTPUTS);
  for (int i = 0; i < fused->numOutputs; i++) {
    // output_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i + roff].region.get_index_space());
    output_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->output_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  roff += fused->numOutputs;
  for (int i = 0; i < fused->numInputs; i++) {
    // input_grad_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i + roff].region.get_index_space());
    input_grad_accessor[i] =
        helperGetGenericTensorAccessorRW(fused->input_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
    assert(input_grad_accessor[i].domain == input_accessor[i].domain);
  }
  roff += fused->numInputs;
  for (int i = 0; i < fused->numWeights; i++) {
    // weight_grad_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i + roff].region.get_index_space());
    weight_grad_accessor[i] =
        helperGetGenericTensorAccessorRW(fused->weight_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
    assert(weight_grad_accessor[i].domain.get_volume() ==
           weight_accessor[i].domain.get_volume());
  }
  roff += fused->numWeights;
  for (int i = 0; i < fused->numOutputs; i++) {
    // output_grad_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i + roff].region.get_index_space());
    output_grad_accessor[i] =
        helperGetGenericTensorAccessorRW(fused->output_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
    assert(output_grad_accessor[i].domain == output_accessor[i].domain);
  }
  roff += fused->numOutputs;
  // Assert that all meta share the same dnn/blas handler
  int start = 0;
  for (start = 0; start < fused->numOperators; start++) {
    if (metas->meta[start] != NULL) {
      break;
    }
  }
  for (int op = start + 1; op < fused->numOperators; op++) {
    if (metas->meta[op] != NULL) {
      assert(metas->meta[start]->handle.blas == metas->meta[op]->handle.blas);
      assert(metas->meta[start]->handle.dnn == metas->meta[op]->handle.dnn);
    }
  }

  int ioff = 0, woff = 0, ooff = 0;
  // Domain my_id[MAX_NUM_INPUTS], my_grad_id[MAX_NUM_INPUTS];
  // Domain my_wd[MAX_NUM_WEIGHTS], my_grad_wd[MAX_NUM_WEIGHTS];
  // Domain my_od[MAX_NUM_OUTPUTS], my_grad_od[MAX_NUM_OUTPUTS];
  GenericTensorAccessorR my_input_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorR my_weight_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorR my_output_accessor[MAX_NUM_OUTPUTS];
  GenericTensorAccessorW my_input_grad_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorW my_weight_grad_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorW my_output_grad_accessor[MAX_NUM_OUTPUTS];
  // Do backpropagation in the reverse ordering
  for (int op = 0; op < fused->numOperators; op++) {
    ioff += fused->op_num_inputs[op];
    woff += fused->op_num_weights[op];
    ooff += fused->op_num_outputs[op];
  }

  for (int op = fused->numOperators - 1; op >= 0; op--) {
    ioff -= fused->op_num_inputs[op];
    woff -= fused->op_num_weights[op];
    ooff -= fused->op_num_outputs[op];
    for (int i = 0; i < fused->op_num_inputs[op]; i++) {
      int my_off = fused->op_input_idx[i + ioff];
      if (fused->op_input_source[i + ioff] == SOURCE_INPUT) {
        // my_id[i] = input_domain[my_off];
        // my_ip[i] = input_ptr[my_off];
        my_input_accessor[i] = input_accessor[my_off];
        // my_grad_id[i] = input_grad_domain[my_off];
        // my_grad_ip[i] = input_grad_ptr[my_off];
        my_input_grad_accessor[i] = input_grad_accessor[my_off];
        assert(my_input_grad_accessor[i].domain == my_input_accessor[i].domain);
      } else if (fused->op_input_source[i + ioff] == SOURCE_OUTPUT) {
        // my_id[i] = output_domain[my_off];
        // my_ip[i] = output_ptr[my_off];
        my_input_accessor[i] = output_accessor[my_off];
        // my_grad_id[i] = output_grad_domain[my_off];
        // my_grad_ip[i] = output_grad_ptr[my_off];
        my_input_grad_accessor[i] = output_grad_accessor[my_off];
        assert(my_input_grad_accessor[i].domain == my_input_accessor[i].domain);
      } else {
        assert(false);
      }
    }
    for (int i = 0; i < fused->op_num_weights[op]; i++) {
      assert(fused->op_weight_source[i + woff] == SOURCE_WEIGHT);
      // my_wd[i] = weight_domain[fused->op_weight_idx[i + woff]];
      // my_wp[i] = weight_ptr[fused->op_weight_idx[i + woff]];
      my_weight_accessor[i] = weight_accessor[fused->op_weight_idx[i + woff]];
      // my_grad_wd[i] = weight_grad_domain[fused->op_weight_idx[i + woff]];
      // my_grad_wp[i] = weight_grad_ptr[fused->op_weight_idx[i + woff]];
      my_weight_grad_accessor[i] =
          weight_grad_accessor[fused->op_weight_idx[i + woff]];
      assert(my_weight_grad_accessor[i].domain.get_volume() ==
             my_weight_accessor[i].domain.get_volume());
    }
    for (int i = 0; i < fused->op_num_outputs[op]; i++) {
      assert(fused->op_output_source[i + ooff] == SOURCE_OUTPUT);
      // my_od[i] = output_domain[fused->op_output_idx[i + ooff]];
      // my_op[i] = output_ptr[fused->op_output_idx[i + ooff]];
      my_output_accessor[i] = output_accessor[fused->op_output_idx[i + ooff]];
      // my_grad_od[i] = output_grad_domain[fused->op_output_idx[i + ooff]];
      // my_grad_op[i] = output_grad_ptr[fused->op_output_idx[i + ooff]];
      my_output_grad_accessor[i] =
          output_grad_accessor[fused->op_output_idx[i + ooff]];
      assert(my_output_grad_accessor[i].domain == my_output_accessor[i].domain);
    }
    switch (fused->op_op_type[op]) {
      case OP_BATCHMATMUL: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        Domain out_domain = my_output_accessor[0].domain;
        Domain a_domain = my_input_accessor[0].domain;
        Domain b_domain = my_input_accessor[1].domain;
        // check dims
        int m = b_domain.hi()[0] - b_domain.lo()[0] + 1;
        assert(m == out_domain.hi()[0] - out_domain.lo()[0] + 1);
        int n = a_domain.hi()[1] - a_domain.lo()[1] + 1;
        assert(n == out_domain.hi()[1] - out_domain.lo()[1] + 1);
        int k = a_domain.hi()[0] - a_domain.lo()[0] + 1;
        assert(k == b_domain.hi()[1] - b_domain.lo()[1] + 1);
        assert(a_domain.get_dim() == b_domain.get_dim());
        assert(a_domain.get_dim() == out_domain.get_dim());
        int batch = 1;
        for (int i = 2; i < a_domain.get_dim(); i++) {
          int dim_size = a_domain.hi()[i] - a_domain.lo()[i] + 1;
          assert(dim_size == b_domain.hi()[i] - b_domain.lo()[i] + 1);
          assert(dim_size == out_domain.hi()[i] - out_domain.lo()[i] + 1);
          batch *= dim_size;
        }
        BatchMatmulPerDeviceState *meta =
            (BatchMatmulPerDeviceState *)metas->meta[op];
        Kernels::BatchMatmul::backward_kernel(
            meta,
            (float const *)my_output_accessor[0].get_float_ptr(),
            (float const *)my_output_grad_accessor[0].get_float_ptr(),
            (float const *)my_input_accessor[0].get_float_ptr(),
            (float *)my_input_grad_accessor[0].get_float_ptr(),
            (float const *)my_input_accessor[1].get_float_ptr(),
            (float *)my_input_grad_accessor[1].get_float_ptr(),
            (float *)nullptr,
            m,
            n,
            k,
            batch);
        break;
      }
      case OP_BATCHNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_dim() == 5);
        assert(my_weight_accessor[0].domain.get_dim() == 2);
        assert(my_weight_accessor[1].domain.get_dim() == 2);
        assert(my_output_accessor[0].domain.get_dim() == 5);
        BatchNormPerDeviceState *m = (BatchNormPerDeviceState *)metas->meta[op];
        BatchNorm::backward_kernel(
            m,
            (float const *)my_input_accessor[0].get_float_ptr(),
            (float *)my_output_grad_accessor[0].get_float_ptr(),
            (float const *)my_output_accessor[0].get_float_ptr(),
            (float *)my_input_grad_accessor[0].get_float_ptr(),
            (float const *)my_weight_accessor[0].get_float_ptr(),
            (float *)my_weight_grad_accessor[0].get_float_ptr(),
            (float *)my_weight_grad_accessor[1].get_float_ptr(),
            my_output_accessor[0].domain.get_volume());
        break;
      }
      case OP_CONCAT: {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        ConcatPerDeviceState *m = (ConcatPerDeviceState *)metas->meta[op];
        int num_inputs = fused->op_num_inputs[op];
        Kernels::Concat::backward_kernel(m,
                                         my_output_grad_accessor[0],
                                         my_input_grad_accessor,
                                         num_inputs,
                                         m->legion_axis);
        break;
      }
      case OP_CONV2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_dim() == 5);
        assert(my_weight_accessor[0].domain.get_dim() == 5);
        assert(my_output_accessor[0].domain.get_dim() == 5);
        Conv2DPerDeviceState *m = (Conv2DPerDeviceState *)metas->meta[op];
        Kernels::Conv2D::backward_kernel(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_weight_accessor[0].get_float_ptr(),
            my_weight_grad_accessor[0].get_float_ptr(),
            my_weight_grad_accessor[1].get_float_ptr());
        break;
      }
      case OP_DROPOUT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        DropoutPerDeviceState *m = (DropoutPerDeviceState *)metas->meta[op];
        Kernels::Dropout::backward_kernel(
            m,
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr());
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_DIV:
      case OP_EW_MAX:
      case OP_EW_MIN: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain == my_input_accessor[1].domain);
        assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        ElementBinaryPerDeviceState *m =
            (ElementBinaryPerDeviceState *)metas->meta[op];
        Kernels::ElementBinary::backward_kernel(
            m,
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_accessor[0].get_float_ptr(),
            my_input_accessor[1].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[1].get_float_ptr());
        break;
      }
      case OP_EMBEDDING: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        EmbeddingPerDeviceState *m = (EmbeddingPerDeviceState *)metas->meta[op];
        assert(my_input_accessor[0].data_type == DT_INT64);
        int in_dim, out_dim, effective_batch_size;
        if (m->aggr == AGGR_MODE_NONE) {
          in_dim = 1;
          out_dim = my_output_grad_accessor[0].domain.hi()[0] -
                    my_output_grad_accessor[0].domain.lo()[0] + 1;
          effective_batch_size =
              my_output_grad_accessor[0].domain.get_volume() / out_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].domain.get_volume());
        } else {
          in_dim = my_input_accessor[0].domain.hi()[0] -
                   my_input_accessor[0].domain.lo()[0] + 1;
          out_dim = my_output_grad_accessor[0].domain.hi()[0] -
                    my_output_grad_accessor[0].domain.lo()[0] + 1;
          effective_batch_size =
              my_output_grad_accessor[0].domain.get_volume() / out_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].domain.get_volume());
        }
        Kernels::Embedding::backward_kernel(m,
                                            my_input_accessor[0],
                                            my_output_grad_accessor[0],
                                            my_weight_grad_accessor[0],
                                            in_dim,
                                            out_dim,
                                            effective_batch_size);
        break;
      }
      case OP_LINEAR: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        Domain kernel_domain = my_weight_accessor[0].domain;
        int in_dim = kernel_domain.hi()[0] - kernel_domain.lo()[0] + 1;
        int out_dim = kernel_domain.hi()[1] - kernel_domain.lo()[1] + 1;
        int batch_size = my_input_accessor[0].domain.get_volume() / in_dim;
        assert(my_output_accessor[0].domain.get_volume() ==
               out_dim * batch_size);
        assert(my_input_accessor[0].domain.get_volume() == in_dim * batch_size);
        float *bias_grad_ptr = nullptr;
        if (fused->op_num_weights[op] == 2) {
          assert(my_weight_accessor[1].domain.get_volume() == out_dim);
          bias_grad_ptr = my_weight_grad_accessor[1].get_float_ptr();
        } else {
          assert(fused->op_num_weights[op] == 1);
        }
        LinearPerDeviceState *m = (LinearPerDeviceState *)metas->meta[op];
        Kernels::Linear::backward_kernel(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_weight_accessor[0].get_float_ptr(),
            my_weight_grad_accessor[0].get_float_ptr(),
            bias_grad_ptr,
            in_dim,
            out_dim,
            batch_size);
        break;
      }
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_ELU: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        ElementUnaryPerDeviceState *m =
            (ElementUnaryPerDeviceState *)metas->meta[op];
        Kernels::ElementUnary::backward_kernel(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_accessor[0].domain.get_volume());
        break;
      }
      case OP_POOL2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        // assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        Pool2DPerDeviceState *m = (Pool2DPerDeviceState *)metas->meta[op];
        Kernels::Pool2D::backward_kernel(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr());
        break;
      }
      case OP_FLAT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].domain.get_volume() ==
               my_output_grad_accessor[0].domain.get_volume());
        Kernels::Flat::backward_kernel(
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].domain.get_volume());
        break;
      }
      case OP_RESHAPE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].domain.get_volume() ==
               my_output_grad_accessor[0].domain.get_volume());
        Kernels::Reshape::backward_kernel(
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].domain.get_volume());
        break;
      }
      case OP_TRANSPOSE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].domain.get_volume() ==
               my_output_grad_accessor[0].domain.get_volume());
        TransposePerDeviceState *m = (TransposePerDeviceState *)metas->meta[op];
        Kernels::Transpose::backward_kernel(
            m,
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].domain,
            my_output_grad_accessor[0].domain);
        break;
      }
      default:
        assert(false && "Fusion currently does not support type");
    }
  }
  assert(ioff == 0);
  assert(woff == 0);
  assert(ooff == 0);
  // for (int i = 0; i < fused->numWeights; i++)
  //   print_tensor<float>(weight_grad_ptr[i],
  //   weight_grad_domain[i].get_volume(), "[Fused:backward:weight_grad]");
  // for (int i = 0; i < fused->numInputs; i++)
  //   print_tensor<float>(input_grad_ptr[i], input_grad_domain[i].get_volume(),
  //   "[Fused:backward:input_grad]");
  // for (int i = 0; i < fused->numOutputs; i++)
  //   print_tensor<float>(output_grad_ptr[i],
  //   output_grad_domain[i].get_volume(), "[Fused:backward:output_grad]");
}

}; // namespace FlexFlow
