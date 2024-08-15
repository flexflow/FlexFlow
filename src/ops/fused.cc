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

#include "flexflow/ops/fused.h"
#include "flexflow/model.h"
#include "flexflow/ops/batch_matmul.h"
#include "flexflow/ops/batch_norm.h"
#include "flexflow/ops/concat.h"
#include "flexflow/ops/dropout.h"
#include "flexflow/ops/element_binary.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/flat.h"
#include "flexflow/ops/pool_2d.h"
#include "flexflow/ops/reshape.h"
#include "flexflow/ops/transpose.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::LogicalPartition;
using Legion::LogicalRegion;
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
  numInputs = 0;
  for (int i = 0; i < op->numInputs; i++) {
    bool found = false;
    // we also need to check region duplicate for the first op in a fused op
    // (e.g., MHA)
    for (int j = 0; j < numInputs; j++) {
      if (inputs[j]->region == op->inputs[i]->region) {
        // This input is one of my inputs
        assert(!found);
        assert(inputs[j]->region != LogicalRegion::NO_REGION);
        op_input_source[i] = SOURCE_INPUT;
        op_input_idx[i] = j;
        found = true;
        break;
      }
    }
    if (found) {
      // do nothing
    } else {
      inputs[numInputs] = op->inputs[i];
      input_data_types[numInputs] = op->inputs[i]->data_type;
      op_input_source[i] = SOURCE_INPUT;
      op_input_idx[i] = numInputs;
      numInputs++;
    }
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
  op_num_inputs[0] = op->numInputs;
  op_num_weights[0] = op->numWeights;
  op_num_outputs[0] = op->numOutputs;
  op_op_type[0] = op->op_type;
  operators[0] = op;
  layer_guid = op->layer_guid;
  // for (int i = 0; i < numInputs; i++) {
  //   op_input_source[i] = SOURCE_INPUT;
  //   op_input_idx[i] = i;
  // }
  for (int i = 0; i < numWeights; i++) {
    op_weight_source[i] = SOURCE_WEIGHT;
    op_weight_idx[i] = i;
  }
  for (int i = 0; i < numOutputs; i++) {
    op_output_source[i] = SOURCE_OUTPUT;
    op_output_idx[i] = i;
  }
}

bool FusedOp::use_same_regions(
    ParallelTensor const source_tensor,
    ParallelTensor const target_tensor,
    std::unordered_map<ParallelTensor, std::vector<ParallelTensor>>
        *pt_mapping) {
  if (pt_mapping == nullptr) {
    return (source_tensor->region == target_tensor->region);
  } else {
    assert(pt_mapping->find(source_tensor) != pt_mapping->end());
    assert(pt_mapping->find(target_tensor) != pt_mapping->end());
    std::vector<ParallelTensor> const &source_mapped_tensor_vector =
        (*pt_mapping)[source_tensor];
    std::vector<ParallelTensor> const &target_mapped_tensor_vector =
        (*pt_mapping)[target_tensor];
    assert(source_mapped_tensor_vector.size() ==
           target_mapped_tensor_vector.size());
    bool same_region = source_mapped_tensor_vector[0]->region ==
                               target_mapped_tensor_vector[0]->region
                           ? true
                           : false;
    // Same that the two vectors use the exact same regions
    if (same_region) {
      for (size_t i = 0; i < source_mapped_tensor_vector.size(); i++) {
        assert(source_mapped_tensor_vector[i]->region ==
               target_mapped_tensor_vector[i]->region);
      }
    }
    return same_region;
  }
}

bool FusedOp::add_operator(
    FFModel &model,
    Op *op,
    std::unordered_map<ParallelTensor, std::vector<ParallelTensor>>
        *pt_mapping) {
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
  // Cannot fuse parallel operators (except allreduce) since they have different
  // paralel_is in forward and backward
  assert(!op->is_parallel_op() || op->op_type == OP_ALLREDUCE);
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
    assert(false);
    return false;
  }
  if (numOperators + 1 > MAX_NUM_FUSED_OPERATORS) {
    fprintf(
        stderr,
        "Reach to the fusion limit. Consider increase MAX_NUM_FUSED_OPERATORS");
    assert(false);
    return false;
  }
  // Set inputs
  for (int i = 0; i < op->numInputs; i++) {
    bool found = false;
    for (int j = 0; j < numInputs; j++) {
      if (use_same_regions(inputs[j], op->inputs[i], pt_mapping)) {
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
      if (use_same_regions(outputs[j], op->inputs[i], pt_mapping) && (!found)) {
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
      // pt_mapping does not apply to weights
      if (pt_mapping != nullptr) {
        assert(pt_mapping->find(weights[j]) == pt_mapping->end());
        assert(pt_mapping->find(op->weights[i]) == pt_mapping->end());
      }
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
      if (use_same_regions(outputs[j], op->outputs[i], pt_mapping)) {
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
      meta[idx++] = fm.get_result<OpMeta *>(*it);                              \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

void FusedOp::init_inference(FFModel const &ff,
                             std::vector<ParallelTensor> const &batch_inputs,
                             std::vector<ParallelTensor> const &batch_outputs,
                             MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = batch_outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Call init methods in individual operators
  Domain domain = runtime->get_index_space_domain(ctx, parallel_is);
  int ioff = 0, ooff = 0;
  for (int op = 0; op < numOperators; op++) {
    // prepare batch_inputs, batch_outputs for operators[op]
    std::vector<ParallelTensor> my_batch_inputs;
    std::vector<ParallelTensor> my_batch_outputs;
    for (int i = 0; i < op_num_inputs[op]; i++) {
      int my_off = op_input_idx[i + ioff];
      if (op_input_source[i + ioff] == SOURCE_INPUT) {
        assert(my_off < batch_inputs.size());
        my_batch_inputs.push_back(batch_inputs[my_off]);
      } else if (op_input_source[i + ioff] == SOURCE_OUTPUT) {
        assert(my_off < batch_outputs.size());
        my_batch_inputs.push_back(batch_outputs[my_off]);
      } else {
        assert(false);
      }
    }
    for (int i = 0; i < op_num_outputs[op]; i++) {
      int my_off = op_output_idx[i + ooff];
      assert(op_output_source[i + ooff] == SOURCE_OUTPUT);
      assert(my_off < batch_outputs.size());
      my_batch_outputs.push_back(batch_outputs[my_off]);
    }
    ioff += op_num_inputs[op];
    ooff += op_num_outputs[op];
    operators[op]->init_inference(ff, my_batch_inputs, my_batch_outputs, mv);
    for (size_t j = 0; j < domain.get_volume(); j++) {
      fused_meta[j].meta[op] =
          operators[op]->inference_meta[my_batch_outputs[0]][j];
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
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  size_t machine_view_hash = view->hash();
  IndexLauncher launcher(FUSEDOP_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(FusedOp)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    int idx = 0;                                                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      inference_meta[batch_outputs[0]][idx++] = fm.get_result<OpMeta *>(*it);  \
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

FutureMap FusedOp::inference(
    FFModel const &ff,
    /* Reserved: BatchConfig Updated */ BatchConfigFuture const &bc,
    std::vector<ParallelTensor> const &batch_inputs,
    std::vector<ParallelTensor> const &batch_outputs,
    MachineView const *mv) {
  // Set iter_config
  iter_config = ff.iter_config;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_inference(ff, argmap, batch_outputs[0]);
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  size_t machine_view_hash = view->hash();
  // bc is one of BatchConfig, TreeVerifyBatchConfig, and BeamSearchBatchConfig
  // so we transfer the maximum of them
  // size_t batch_config_size =
  //    std::max(sizeof(TreeVerifyBatchConfig), sizeof(BeamSearchBatchConfig));
  IndexLauncher launcher(FUSEDOP_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  int offset = 0;
  for (int i = 0; i < numInputs; i++) {
    assert(inputs[i]->part != LogicalPartition::NO_PART);
    assert(inputs[i]->region != LogicalRegion::NO_REGION);
    launcher.add_region_requirement(RegionRequirement(batch_inputs[i]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      batch_inputs[i]->region));
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
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[i]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[i]->region));
    launcher.add_field(offset + i, FID_DATA);
  }
  return runtime->execute_index_space(ctx, launcher);
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

}; // namespace FlexFlow
