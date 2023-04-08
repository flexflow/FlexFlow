/* Copyright 2022 CMU, Stanford, Facebook, LANL
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

#include "flexflow/inference.h"
#include "flexflow/parallel_ops/parallel_op.h"

namespace FlexFlow {

using namespace Legion;

InferenceManager::InferenceManager(FFModel *_model,
                                   int _max_num_requests_per_batch,
                                   int _max_num_inflight_batches)
    : model(_model), max_num_requests_per_batch(_max_num_requests_per_batch),
      max_num_inflight_batches(_max_num_inflight_batches) {
  // populate array of valid single-device machine views
  num_devices = model->config.workersPerNode * model->config.numNodes;
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

void InferenceManager::compile_model_and_allocate_buffer(void) {
  std::vector<MetricsType> metrics;
  model->config.batchSize = max_num_requests_per_batch;
  model->compile(
      LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE, metrics, COMP_MODE_INFERENCE);
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  for (auto const &op : model->operators) {
    // Skip weight operators
    if (op->op_type == OP_WEIGHT) {
      continue;
    }
    for (int i = 0; i < op->numOutputs; i++) {
      ParallelTensor pt_base = op->outputs[i];
      assert(tensor_buffer.find(pt_base) == tensor_buffer.end());
      std::vector<ParallelTensor> list;
      for (int j = 0; j < max_num_inflight_batches; j++) {
        // Copy the metadata from pt_base to pt
        ParallelTensor pt = new ParallelTensorBase(*pt_base);
        pt->region =
            runtime->create_logical_region(ctx,
                                           pt_base->region.get_index_space(),
                                           pt_base->region.get_field_space());
        pt->part = runtime->get_logical_partition(
            ctx, pt->region, pt_base->part.get_index_partition());
        list.push_back(pt);
      }
      tensor_buffer[pt_base] = list;
    }
  }
  // Set machine_view for batch_tensors in the tensor_buffer
  for (int batch_index = 0; batch_index < max_num_inflight_batches;
       batch_index++) {
    int expert_device_index = 0;
    int device_index = batch_index % num_devices;
    for (size_t o = 0; o < model->operators.size(); o++) {
      Op *op = model->operators[o];
      if (op->op_type == OP_WEIGHT) {
        continue;
      }
      MachineView *view;
      if (op->op_type == OP_EXPERTS) {
        view = get_machine_view(expert_device_index);
        // view = &machine_views[expert_device_index];
        expert_device_index = (expert_device_index + 1) % num_devices;
      } else {
        // pick mv w startdeviceid = device_index
        // view = &machine_views[device_index];
        view = get_machine_view(device_index);
      }
      for (int i = 0; i < op->numOutputs; i++) {
        tensor_buffer[op->outputs[i]][batch_index]->machine_view = *view;
        Domain part_domain =
            runtime->get_index_space_domain(ctx, op->outputs[i]->parallel_is);
        assert(view->get_domain() == part_domain);
      }
    }
  }
}

void InferenceManager::init_operators_inference() {
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

MachineView *InferenceManager::get_machine_view(int mv_id) {
  assert(mv_id >= 0 && mv_id < machine_views.size());
  return &machine_views[mv_id];
}

FutureMap InferenceManager::inference(int index, BatchConfig const &bc) {
  // We currently assume that the index-th batch will be placed
  // on the device_index-th device (except for the experts layers)
  int batch_index = index % max_num_inflight_batches;
  FutureMap fm;
  for (size_t o = 0; o < model->operators.size(); o++) {
    Op *op = model->operators[o];
    if (op->op_type == OP_WEIGHT || op->op_type == OP_INPUT) {
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

}; // namespace FlexFlow
