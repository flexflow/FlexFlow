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
#define MAX_NUM_SAMPLES 4196

using namespace Legion;
using namespace std;

struct CandleConfig {
  CandleConfig(void);
  vector<int> dense_layers, dense_feature_layers;
  map<string, int> feature_shapes;
  map<string, string> input_features;
  std::string dataset_path;
};

class DataLoader {
public:
  DataLoader(FFModel& ff, const CandleConfig& candle,
             const std::vector<Tensor>& _all_inputs,
             Tensor _label);
  static void load_input(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  static void load_entire_dataset(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx,
                                  Runtime* runtime);
  void next_batch(FFModel&);
  void reset(void);
public:
  int num_samples, next_index;
  std::vector<Tensor> full_inputs, batch_inputs;
  Tensor full_label, batch_label;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};

