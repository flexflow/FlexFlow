/* Copyright 2023 CMU, Stanford, Facebook, LANL
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

#include "flexflow/ffconst_utils.h"
#include "flexflow/graph.h"
#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/ops/fused.h"
#include "flexflow/ops/noop.h"
#include "flexflow/parallel_ops/parallel_op.h"

namespace FlexFlow {

using namespace Legion;

LegionRuntime::Logger::Category log_inf_mgr("InferenceManager");
LegionRuntime::Logger::Category log_offload("Offloading");

InferenceManager::InferenceManager(FFConfig const &_config,
                                   int _max_num_tokens_per_batch,
                                   int _max_num_inflight_batches)
    : ff_config(_config), max_num_tokens_per_batch(_max_num_tokens_per_batch),
      max_num_inflight_batches(_max_num_inflight_batches) {
  // populate array of valid single-device machine views
  num_devices = ff_config.workersPerNode * ff_config.numNodes;
  for (int i = 0; i < num_devices; i++) {
    MachineView view;
    view.device_type = MachineView::GPU;
    view.ndims = 1;
    view.dim[0] = 1;
    view.stride[0] = 0;
    view.start_device_id = i;
    machine_views.push_back(view);
  }
}

bool parallel_tensor_list_overlaps(std::vector<ParallelTensor> const &list1,
                                   std::vector<ParallelTensor> const &list2) {
  for (auto const &pt1 : list1) {
    for (auto const &pt2 : list2) {
      if (pt1 == pt2) {
        return true;
      }
    }
  }
  return false;
}

void InferenceManager::compile_model_and_allocate_buffer(
    FFModel *model,
    std::unordered_map<Tensor, std::vector<MachineView>> const
        &tensor_mapping) {
  model->config.batchSize = max_num_tokens_per_batch;
  model->compile_inference();
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;

  std::unordered_map<Op const *, std::vector<MachineView>> mapping;
  for (auto const &it : tensor_mapping) {
    ParallelTensor pt;
    model->get_parallel_tensor_from_tensor(it.first, pt);
    assert(pt->owner_op != nullptr);
    mapping[pt->owner_op] = it.second;
  }
  for (int op_idx = 0; op_idx < model->operators.size(); op_idx++) {
    Op const *op = model->operators[op_idx];
    // Skip weight operators
    if (op->op_type == OP_WEIGHT) {
      continue;
    }
    // Get machine views
    std::vector<MachineView> machine_views;
    if (mapping.find(op) != mapping.end()) {
      machine_views = mapping[op];
      assert(machine_views.size() == max_num_inflight_batches);
    } else {
      // Mapping the current operator using the same machine
      // view as the inputs
      assert(op->numInputs > 0);
      for (int j = 0; j < max_num_inflight_batches; j++) {
        MachineView mv = tensor_buffer[op->inputs[0]][j]->machine_view;
        for (int k = 1; k < op->numInputs; k++) {
          if (mv != tensor_buffer[op->inputs[k]][j]->machine_view) {
            fprintf(stderr,
                    "[Warning] a potentially unnecessary "
                    " inter-GPU copy of size %zu\n",
                    op->inputs[k]->get_volume());
            // Heuristics: we use the mv with a larger start_device_id
            // to promote load balancing
            if (mv.start_device_id <
                tensor_buffer[op->inputs[k]][j]->machine_view.start_device_id) {
              mv = tensor_buffer[op->inputs[k]][j]->machine_view;
            }
          }
        }
        machine_views.push_back(mv);
      }
      assert(machine_views.size() == max_num_inflight_batches);
    }
    for (int i = 0; i < op->numOutputs; i++) {
      ParallelTensor pt_base = op->outputs[i];
      assert(tensor_buffer.find(pt_base) == tensor_buffer.end());
      std::vector<ParallelTensor> list;
      bool found_parallel_tensor = false;
      if (model->config.cpu_offload) {
        for (auto const &pre_pt : tensor_buffer) {
          bool used_by_future_operator = false;
          bool used_by_current_operator = false;
          if (pre_pt.first->get_shape() != pt_base->get_shape()) {
            // Continue if shape mismatches
            continue;
          }
          // Check that pt cannot be used as an input to the current operator
          for (int j = 0; j < op->numInputs; j++) {
            if (parallel_tensor_list_overlaps(tensor_buffer[op->inputs[j]],
                                              pre_pt.second)) {
              used_by_current_operator = true;
            }
          }
          // Check that pt cannot be used by any subsequent operators
          for (int op_idx2 = op_idx; op_idx2 < model->operators.size();
               op_idx2++) {
            Op const *op2 = model->operators[op_idx2];
            for (int j = 0; j < op2->numInputs; j++) {
              if (tensor_buffer.find(op2->inputs[j]) != tensor_buffer.end()) {
                if (parallel_tensor_list_overlaps(tensor_buffer[op2->inputs[j]],
                                                  pre_pt.second)) {
                  used_by_future_operator = true;
                }
              }
            }
          }
          if (!used_by_future_operator && !used_by_current_operator) {
            found_parallel_tensor = true;
            list = pre_pt.second;
          }
        }
        if (!found_parallel_tensor) {
          log_offload.print(
              "Cannot find a previous tensor for operator(%d) output_idx(%d)",
              op_idx,
              i);
        }
      }
      if (!found_parallel_tensor) {
        for (int j = 0; j < max_num_inflight_batches; j++) {
          // Copy the metadata from pt_base to pt
          ParallelTensor pt = new ParallelTensorBase(*pt_base);
          pt->region =
              runtime->create_logical_region(ctx,
                                             pt_base->region.get_index_space(),
                                             pt_base->region.get_field_space());
          pt->part = runtime->get_logical_partition(
              ctx, pt->region, pt_base->part.get_index_partition());
          pt->machine_view = machine_views[j];
          Domain part_domain =
              runtime->get_index_space_domain(ctx, pt_base->parallel_is);
          assert(pt->machine_view.get_domain() == part_domain);
          list.push_back(pt);
        }
      }
      assert(tensor_buffer.find(pt_base) == tensor_buffer.end());
      tensor_buffer[pt_base] = list;
    }
  }
}

void InferenceManager::init_operators_inference(FFModel *model) {
  for (int batch_index = 0; batch_index < max_num_inflight_batches;
       batch_index++) {
    int expert_device_index = 0;
    int device_index = batch_index % num_devices;
    for (size_t o = 0; o < model->operators.size(); o++) {
      Op *op = model->operators[o];
      if (op->op_type == OP_WEIGHT) {
        continue;
      }
      std::vector<ParallelTensor> inputs(op->numInputs);
      std::vector<ParallelTensor> outputs(op->numOutputs);
      for (int i = 0; i < op->numInputs; i++) {
        assert(op->inputs[i] != nullptr);
        assert(op->inputs[i]->parallel_is != IndexSpace::NO_SPACE);
        assert(tensor_buffer[op->inputs[i]].size() > batch_index);
        inputs[i] = tensor_buffer[op->inputs[i]][batch_index];
        assert(inputs[i]->parallel_is != IndexSpace::NO_SPACE);
      }
      assert(op->numOutputs > 0);
      for (int i = 0; i < op->numOutputs; i++) {
        assert(op->outputs[i] != nullptr);
        assert(op->outputs[i]->parallel_is != IndexSpace::NO_SPACE);
        assert(tensor_buffer[op->outputs[i]].size() > batch_index);
        outputs[i] = tensor_buffer[op->outputs[i]][batch_index];
        if (i > 0) {
          assert(outputs[0]->machine_view == outputs[i]->machine_view);
        }
        assert(outputs[i]->parallel_is != IndexSpace::NO_SPACE);
      }
      if (op->is_parallel_op()) {
        ((ParallelOp *)op)
            ->create_input_partition_inference(*model, inputs, outputs);
      }
      op->init_inference(*model, inputs, outputs);
    }
  }
}

// Deprecated API
MachineView *InferenceManager::get_machine_view(int mv_id) {
  assert(false);
  assert(mv_id >= 0 && mv_id < machine_views.size());
  return &machine_views[mv_id];
}

FutureMap InferenceManager::inference(FFModel *model,
                                      int index,
                                      BatchConfig const &bc) {
  log_inf_mgr.print("mode(%d) num_active_tokens(%d) num_active_requests(%d)",
                    bc.get_mode(),
                    bc.num_active_tokens(),
                    bc.num_active_requests());

  assert(bc.num_active_tokens() > 0 && bc.num_active_requests() > 0);
  // We currently assume that the index-th batch will be placed
  // on the device_index-th device (except for the experts layers)
  int batch_index = index % max_num_inflight_batches;
  FutureMap fm;
  bool found_input_operator = false;
  for (size_t o = 0; o < model->operators.size(); o++) {
    Op *op = model->operators[o];
    if (op->op_type == OP_WEIGHT) {
      continue;
    }
    if (op->op_type == OP_INPUT) {
      // FIXME: this is a hack, should be replace with an input ParallelTensor
      if (found_input_operator) {
        // there is another input for position embedding;
        // now only used in opt model, this input should be init after token
        // input.
        assert(op->numOutputs == 1);
        ParallelTensor pt = tensor_buffer[op->outputs[0]][batch_index];
        load_positions(bc, pt);
      } else {
        found_input_operator = true;
        assert(op->numOutputs == 1);
        ParallelTensor pt = tensor_buffer[op->outputs[0]][batch_index];
        load_input_tokens_from_batch_config(bc, pt);
      }
    }

    std::vector<ParallelTensor> inputs(op->numInputs);
    std::vector<ParallelTensor> outputs(op->numOutputs);
    for (int i = 0; i < op->numInputs; i++) {
      assert(op->inputs[i] != nullptr);
      assert(op->inputs[i]->parallel_is != IndexSpace::NO_SPACE);
      assert(tensor_buffer[op->inputs[i]].size() > batch_index);
      inputs[i] = tensor_buffer[op->inputs[i]][batch_index];
      assert(inputs[i]->parallel_is != IndexSpace::NO_SPACE);
    }
    for (int i = 0; i < op->numOutputs; i++) {
      assert(op->outputs[i] != nullptr);
      assert(op->outputs[i]->parallel_is != IndexSpace::NO_SPACE);
      if (op->op_type == OP_INPUT &&
          tensor_buffer[op->outputs[i]].size() == 0) {
        continue;
      }
      assert(tensor_buffer[op->outputs[i]].size() > batch_index);
      outputs[i] = tensor_buffer[op->outputs[i]][batch_index];
      assert(outputs[i]->parallel_is != IndexSpace::NO_SPACE);
    }
    fm = op->inference(*model, bc, inputs, outputs);
  }
  return fm;
};

void InferenceManager::load_input_tokens_from_batch_config(
    BatchConfig const &bc, ParallelTensor const input) {
  Context ctx = ff_config.lg_ctx;
  Runtime *runtime = ff_config.lg_hlr;
  size_t machine_view_hash = input->machine_view.hash();
  ArgumentMap argmap;
  IndexLauncher launcher(
      RM_LOAD_TOKENS_TASK_ID,
      input->parallel_is,
      TaskArgument(
          &bc, std::max(sizeof(BeamSearchBatchConfig), sizeof(BatchConfig))),
      argmap,
      Predicate::TRUE_PRED,
      false /*must*/,
      0 /*mapper_id*/,
      machine_view_hash);
  launcher.add_region_requirement(RegionRequirement(
      input->part, 0 /*projection id*/, WRITE_ONLY, EXCLUSIVE, input->region));
  launcher.add_field(0, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void InferenceManager::load_positions(BatchConfig const &bc,
                                      ParallelTensor position_input) {
  Context ctx = ff_config.lg_ctx;
  Runtime *runtime = ff_config.lg_hlr;
  size_t machine_view_hash = position_input->machine_view.hash();
  ArgumentMap argmap;
  IndexLauncher launcher(
      RM_LOAD_POSITION_TASK_ID,
      position_input->parallel_is,
      TaskArgument(
          &bc, std::max(sizeof(BeamSearchBatchConfig), sizeof(BatchConfig))),
      argmap,
      Predicate::TRUE_PRED,
      false /*must*/,
      0 /*mapper_id*/,
      machine_view_hash);
  launcher.add_region_requirement(RegionRequirement(position_input->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    position_input->region));
  launcher.add_field(0, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void FFModel::compile_inference() {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  config.computationMode = COMP_MODE_INFERENCE;
  {
    fprintf(
        stderr,
        "Note: inference currently only supports data/pipeline parallel.\n");
  }
  create_operators_from_layers();
  // Launch the graph optimize task
  {
    FFModel *model = this;
    TaskLauncher launcher(GRAPH_OPTIMIZE_TASK_ID,
                          TaskArgument(&model, sizeof(FFModel *)));
    Future future = runtime->execute_task(ctx, launcher);

    PCG::GraphOptimalViewSerialized ret =
        future.get_result<PCG::GraphOptimalViewSerialized>();
    Deserializer dez(ret.data, ret.total_bytes);
    // Reconstruct operators
    PCG::Graph *best_graph = new PCG::Graph(this);
    std::unordered_map<PCG::Node, MachineView> optimal_views;
    deserialize_graph_optimal_view(dez, best_graph, optimal_views);
    operators.clear();
    convert_graph_to_operators(best_graph, optimal_views);
    best_graph->print_dot();
    delete best_graph;
    for (auto const &layer : layers) {
      // map inputs to parallel tensor
      if (layer->op_type == OP_INPUT) {
        Tensor tensor = layer->outputs[0];
        ParallelTensor parallel_tensor = nullptr;
        for (auto const &op : operators) {
          if (op->op_type == OP_INPUT) {
            NoOp *noop = (NoOp *)op;
            if (noop->input_tensor_guid == tensor->tensor_guid) {
              parallel_tensor = op->outputs[0];
            }
          }
        }
        assert(parallel_tensor != nullptr);
        tensor->parallel_tensor = parallel_tensor;
      }
      // map weights to parallel_tensor
      for (int i = 0; i < layer->numWeights; i++) {
        assert(layer->weights[i] != nullptr);
        Tensor weight = layer->weights[i];
        ParallelTensor parallel_weight = nullptr;
        for (auto const &op : operators) {
          if (op->layer_guid == layer->layer_guid) {
            assert(op->op_type == layer->op_type);
            assert(op->numWeights == layer->numWeights);
            parallel_weight = op->weights[i];
          }
        }
        assert(parallel_weight != nullptr);
        weight->parallel_tensor = parallel_weight;
      }
    }
  }
  loss_op = nullptr;
  metrics_op = nullptr;
  // Perform inplace optimizations
  if (config.enable_inplace_optimizations) {
    for (size_t l = 1; l < operators.size(); l++) {
      if (operators[l]->can_inplace_output()) {
        // Assume outputs[0] is inplace with inputs[0]
        assert(operators[l]->numOutputs == 1);
        if (operators[l]->inputs[0]->owner_op != NULL) {
          // int dim1 = operators[l]->outputs[0]->num_dims;
          // int dim2 = operators[l]->inputs[0]->num_dims;
          MachineView view1 = operators[l]->outputs[0]->machine_view;
          MachineView view2 = operators[l]->inputs[0]->machine_view;
          if (view1 == view2) {
            // Check no others also need operators[l]->inputs[0]
            bool found = false;
            for (size_t i = 0; i < operators.size(); i++) {
              if (i == l) {
                continue;
              }
              for (int j = 0; j < operators[i]->numInputs; j++) {
                if ((operators[i]->inputs[j]->owner_op ==
                     operators[l]->inputs[0]->owner_op) &&
                    (operators[i]->inputs[j]->owner_idx ==
                     operators[l]->inputs[0]->owner_idx)) {
                  found = true;
                }
              }
            }
            if (!found) {
              // Perform inplace
              operators[l]->do_inplace_output();
            }
          }
        }
      }
    }
  }

  for (size_t l = 0; l < operators.size(); l++) {
    Op *op = operators[l];

    for (int i = 0; i < op->numInputs; i++) {
      assert(op->inputs[i]->owner_op != NULL);
    }
    for (int i = 0; i < op->numWeights; i++) {
      assert(op->weights[i]->owner_op != NULL);
      assert(op->weights[i]->region != LogicalRegion::NO_REGION);
      parameters.push_back(op->weights[i]);
    }
    op->map_output_tensors(*this);
  }

  // Check correctness
  for (size_t l = 0; l < operators.size(); l++) {
    Op *op = operators[l];
    for (int i = 0; i < op->numOutputs; i++) {
      assert(op->outputs[i]->owner_op == op);
      assert(op->outputs[i]->owner_idx == i);
      assert(op->outputs[i]->parallel_tensor_guid != 0);
    }
  }
  // Perform fusion optimizations
  if (config.perform_fusion) {
    fprintf(stderr, "Applying fusion optimizations during compilation...\n");
    fprintf(stderr, "%zu operators before fusion...\n", operators.size());
    std::vector<Op *> new_operators;
    std::vector<Op *> old_operators = operators;
    while (apply_fusion(operators, new_operators)) {
      for (size_t i = 0; i < new_operators.size(); i++) {
        for (int idx = 0; idx < new_operators[i]->numInputs; idx++) {
          for (size_t j = i + 1; j < new_operators.size(); j++) {
            if (new_operators[i]->inputs[idx]->owner_op == new_operators[j]) {
              assert(false);
            }
          }
        }
      }
      operators = new_operators;
    }
    // Check integrity
    for (size_t l = 0; l < operators.size(); l++) {
      if (operators[l]->op_type == OP_FUSED) {
        FusedOp *fused = (FusedOp *)operators[l];
        int ioff = 0, woff = 0, ooff = 0;
        for (int op = 0; op < fused->numOperators; op++) {
          Op *old_op = fused->operators[op];
          for (int i = 0; i < fused->op_num_inputs[op]; i++) {
            int my_off = fused->op_input_idx[i + ioff];
            if (fused->op_input_source[i + ioff] == FusedOp::SOURCE_INPUT) {
              assert(fused->inputs[my_off]->region ==
                     old_op->inputs[i]->region);
            } else if (fused->op_input_source[i + ioff] ==
                       FusedOp::SOURCE_OUTPUT) {
              assert(fused->outputs[my_off]->region ==
                     old_op->inputs[i]->region);
            } else {
              assert(false);
            }
          }
          for (int i = 0; i < fused->op_num_weights[op]; i++) {
            int my_off = fused->op_weight_idx[i + woff];
            assert(fused->op_weight_source[i + woff] == FusedOp::SOURCE_WEIGHT);
            assert(fused->weights[my_off]->region ==
                   old_op->weights[i]->region);
          }
          for (int i = 0; i < fused->op_num_outputs[op]; i++) {
            int my_off = fused->op_output_idx[i + ooff];
            assert(fused->op_output_source[i + ooff] == FusedOp::SOURCE_OUTPUT);
            assert(fused->outputs[my_off]->region ==
                   old_op->outputs[i]->region);
          }
          ioff += fused->op_num_inputs[op];
          woff += fused->op_num_weights[op];
          ooff += fused->op_num_outputs[op];
        }
      } else {
        bool found = false;
        for (size_t i = 0; i < old_operators.size(); i++) {
          if (old_operators[i] == operators[l]) {
            assert(!found);
            found = true;
          }
        }
        assert(found);
      }
    }
    fprintf(stderr, "%zu operators after fusion...\n", operators.size());
    for (size_t i = 0; i < operators.size(); i++) {
      Op *op = operators[i];
      printf("operator[%zu]: type(%s) guid(%lu)\n",
             i,
             get_operator_type_name(operators[i]->op_type).c_str(),
             operators[i]->op_guid);
      for (int j = 0; j < op->numInputs; j++) {
        LogicalRegion handle = op->inputs[j]->region;
        printf("\tinputs[%d] region(%d,%d,%d)\n",
               j,
               handle.get_index_space().get_id(),
               handle.get_field_space().get_id(),
               handle.get_tree_id());
      }
      for (int j = 0; j < op->numOutputs; j++) {
        LogicalRegion handle = op->outputs[j]->region;
        printf("\toutputs[%d] region(%d,%d,%d)\n",
               j,
               handle.get_index_space().get_id(),
               handle.get_field_space().get_id(),
               handle.get_tree_id());
      }
      for (int j = 0; j < op->numWeights; j++) {
        LogicalRegion handle = op->weights[j]->region;
        printf("\tweights[%d] region(%d,%d,%d)\n",
               j,
               handle.get_index_space().get_id(),
               handle.get_field_space().get_id(),
               handle.get_tree_id());
      }
    }
  }
  for (size_t i = 0; i < operators.size(); i++) {
    Op *op = operators[i];
    printf("operator[%zu]: type(%d)\n", i, operators[i]->op_type);
    for (int j = 0; j < op->numInputs; j++) {
      LogicalRegion handle = op->inputs[j]->region;
      printf("\tinputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
    for (int j = 0; j < op->numOutputs; j++) {
      LogicalRegion handle = op->outputs[j]->region;
      printf("\toutputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
  }
}
}; // namespace FlexFlow
