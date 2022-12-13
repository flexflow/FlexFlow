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

namespace FlexFlow {

using namespace Legion;

InferenceManager::InferenceManager(FFModel *_model,
                                   int _max_num_requests_per_batch,
                                   int _max_num_inflight_batches)
  : model(_model),
    max_num_requests_per_batch(_max_num_requests_per_batch),
    max_num_inflight_batches(_max_num_inflight_batches) {

}

void InferenceManager::compile_model_and_allocate_buffer(void) {
  std::vector<MetricsType> metrics;
  model->config.batchSize = max_num_requests_per_batch;
  model->compile(LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE, metrics, COMP_MODE_INFERENCE);
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  for (const auto& op : model->operators) {
    // Skip weight operators
    if (op->op_type == OP_WEIGHT)
      continue;
    for (int i = 0; i < op->numOutputs; i++) {
      ParallelTensor pt_base = op->outputs[i];
      assert(tensor_buffer.find(pt_base) == tensor_buffer.end());
      std::vector<ParallelTensor> list;
      for (int j = 0; j < max_num_inflight_batches; j++) {
        // Copy the metadata from pt_base to pt
        ParallelTensor pt = new ParallelTensorBase(*pt_base);
        pt->region = runtime->create_logical_region(ctx,
                                                    pt_base->region.get_index_space(),
                                                    pt_base->region.get_field_space());
        pt->part = runtime->get_logical_partition(ctx, pt->region, pt_base->part.get_index_partition());
        list.push_back(pt);
      }
      tensor_buffer[pt_base] = list;
    }
  }
}

void InferenceManager::inference(int index) {
  assert(index < max_num_inflight_batches);
  for (size_t o = 0; o < model->operators.size(); o++) {
    Op* op = model->operators[o];
    std::vector<ParallelTensor> inputs(op->numInputs);
    std::vector<ParallelTensor> outputs(op->numOutputs);
    for (int i = 0; i < op->numInputs; i++)
      inputs[i] = tensor_buffer[op->inputs[i]][index];
    for (int i = 0; i < op->numOutputs; i++)
      outputs[i] = tensor_buffer[op->outputs[i]][index];
    op->inference(*model, inputs, outputs);
  }
};

};
