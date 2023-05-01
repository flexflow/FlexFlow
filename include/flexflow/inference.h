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

#pragma once

#include "flexflow/batch_config.h"
#include "flexflow/model.h"
#include <mutex>

namespace FlexFlow {

class FFModel;

class InferenceManager {
public:
  InferenceManager(FFModel *_model,
                   int max_num_tokens_per_batch,
                   int max_num_inflight_batches);
  void compile_model_and_allocate_buffer(
      FFModel *model,
      std::unordered_map<Tensor, std::vector<MachineView>> const &mapping);
  void init_operators_inference(FFModel *model);
  MachineView *get_machine_view(int mv_id);
  Legion::FutureMap inference(FFModel *model,
                              int index,
                              BatchConfigFuture const &batch_config);
  Legion::FutureMap
      inference(FFModel *model, int index, BatchConfig const &batch_config);
  void load_input_tokens_from_batch_config(BatchConfigFuture const &bc,
                                           ParallelTensor const input);

public:
  std::unordered_map<ParallelTensor, std::vector<ParallelTensor>> tensor_buffer;
  FFModel *model;
  int max_num_tokens_per_batch;
  int max_num_inflight_batches;
  int num_devices;
  std::vector<MachineView> machine_views;
};

struct Request {
  BatchConfig::RequestGuid guid;
  int max_sequence_length;
  std::vector<BatchConfig::TokenId> tokens;
};

class RequestManager {
public:
  using RequestGuid = BatchConfig::RequestGuid;
  using TokenId = BatchConfig::TokenId;
  RequestManager(FFConfig const &config);
  RequestGuid register_new_request(std::vector<TokenId> const &prompt,
                                   int max_sequence_length);
  BatchConfig prepare_next_batch(BatchConfig const &bc,
                                 InferenceResult const &result);
  BatchConfigFuture prepare_next_batch(BatchConfigFuture const &bc,
                                       InferenceResultFuture const &result);
  size_t get_num_processed_requests();
  static void
      load_tokens_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static BatchConfig prepare_next_batch_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

private:
  FFConfig config;
  std::queue<Request> pending_request_queue;
  std::unordered_map<RequestGuid, Request> running_request_queue;
  std::mutex request_queue_mutex;
  RequestGuid next_available_guid;
  size_t num_processed_requests;
};

} // namespace FlexFlow
