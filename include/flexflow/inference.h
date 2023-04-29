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
#include "flexflow/batch_config_v2.h"
#include "flexflow/model.h"

namespace FlexFlow {

class FFModel;

class InferenceManager {
public:
  InferenceManager(FFModel *_model,
                   int max_num_requests_per_batch,
                   int max_num_inflight_batches);
  void compile_model_and_allocate_buffer(void);
  void init_operators_inference();
  MachineView *get_machine_view(int mv_id);
  Legion::FutureMap inference(int index, BatchConfig const &bc);

public:
  std::unordered_map<ParallelTensor, std::vector<ParallelTensor>> tensor_buffer;
  FFModel *model;
  int max_num_requests_per_batch;
  int max_num_inflight_batches;
  int num_devices;
  std::vector<MachineView> machine_views;
};

} // namespace FlexFlow
