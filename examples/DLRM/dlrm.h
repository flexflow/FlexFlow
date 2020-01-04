/* Copyright 2019 Stanford
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

#include "model.h"
#define MAX_NUM_SAMPLES 65536

using namespace Legion;

struct DLRMConfig {
  DLRMConfig(void)
  : sparse_feature_size(2), sigmoid_bot(-1), sigmoid_top(-1),
    embedding_bag_size(1), loss_threshold(0.0f),
    arch_interaction_op("cat"), dataset_path("") {
    embedding_size.push_back(4);
    //embedding_size.push_back(4);
    //embedding_size.push_back(4);
    mlp_bot.push_back(4);
    mlp_bot.push_back(2);
    mlp_top.push_back(8);
    mlp_top.push_back(2);
  }
  int sparse_feature_size, sigmoid_bot, sigmoid_top, embedding_bag_size;
  float loss_threshold;
  std::vector<int> embedding_size, mlp_bot, mlp_top;
  std::string arch_interaction_op, dataset_path;
};

class DataLoader {
public:
  DataLoader(FFModel& ff, const DLRMConfig& dlrm,
             const std::vector<Tensor>& _sparse_inputs,
             Tensor _dense_input, Tensor _label);
  void next_batch(FFModel& ff);
  void shuffle();
  void reset();
  static void load_entire_dataset(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx,
                                  Runtime* runtime);
  static void load_sparse_input(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx,
                                Runtime* runtime);
  static void load_sparse_input_cpu(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx,
                                Runtime* runtime);
  static void load_dense_input(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx,
                               Runtime* runtime);
  static void load_label(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
public:
  int num_samples, next_index;
private:
  std::vector<Tensor> batch_sparse_inputs;
  Tensor full_sparse_input, full_dense_input, batch_dense_input, full_label, batch_label;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};

