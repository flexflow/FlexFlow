/* Copyright 2020 Stanford
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

#include "flexflow/model.h"
#define MAX_NUM_SAMPLES 4196

using namespace Legion;
using namespace std;
using FlexFlow::TensorAccessorR;
using FlexFlow::TensorAccessorW;
using FlexFlow::FID_DATA;

struct AlexNetConfig {
  AlexNetConfig(void) {
    // Set default configurations here
    std::memset(dataset_path, 0, MAX_FILE_LENGTH);
  }
  char dataset_path[MAX_FILE_LENGTH];
};

class DataLoader {
public:
  DataLoader(FlexFlow::FFModel& ff, const AlexNetConfig* alexnet,
             FlexFlow::Tensor _input, FlexFlow::Tensor _label);
  static void load_input(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  static void load_label(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  static void load_entire_dataset(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx,
                                  Runtime* runtime);
  //                                
  void next_batch(FlexFlow::FFModel&);
  void next_label_ubatch(FlexFlow::FFModel&);
  void next_input_ubatch(FlexFlow::FFModel&);
  void reset(void);
  void reset_idx(void);
public:
  int num_samples, next_index, next_input_index, next_label_index;
  FlexFlow::Tensor full_input, batch_input;
  FlexFlow::Tensor full_label, batch_label;
  int input_idx = 0;
  int label_idx = 0;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};

