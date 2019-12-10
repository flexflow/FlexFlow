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

using namespace Legion;
using namespace std;

struct CandleConfig {
  CandleConfig(void) {
    // Set default configurations here
    for (int i = 0; i < 3; i++)
      dense_layers.push_back(1000);
    for (int i = 0; i < 3; i++)
      dense_feature_layers.push_back(1000);
    feature_shapes["dose"] = 1;
    feature_shapes["cell.rnaseq"] = 942;
    feature_shapes["drug.descriptors"] = 5270;
    feature_shapes["drug.fingerprints"] = 2048;
    input_features["dose1"] = "dose";
    input_features["cell.rnaseq"] = "cell.rnaseq";
    input_features["drug1.descriptors"] = "drug.descriptors";
    input_features["drug1.fingerprints"] = "drug.fingerprints";
    input_features["drug2.descriptors"] = "drug.descriptors";
    input_features["drug2.fingerprints"] = "drug.fingerprints";
  }
  vector<int> dense_layers, dense_feature_layers;
  map<string, int> feature_shapes;
  map<string, string> input_features;
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
  static void load_label(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
};
