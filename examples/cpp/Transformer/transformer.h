/* Copyright 2021 Facebook, Stanford
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

struct TransformerConfig {
  TransformerConfig(void);
  int hidden_size, embedding_size, num_heads, num_layers, sequence_length;
};

class DataLoader {
public:
  DataLoader(FFModel& ff, const TransformerConfig& tf,
             const Tensor& _input,
             const Tensor& _label);
  void next_batch(FFModel& ff);
  void reset();
  static void load_entire_dataset(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx,
                                  Runtime* runtime);
  static void load_input(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
public:
  int num_samples, next_index;
private:
  Tensor full_input, batch_input, full_label, batch_label;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};

