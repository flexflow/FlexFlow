/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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
#define MAX_NUM_SAMPLES 1000
#define NUM_SAMPLES 1000
#define TRAIN_SAMPLES 1000
#define TEST_SAMPLES 00000
#define MNIST_DIMS 28 * 28
#define CIFAR_DIMS 3 * 32 * 32
#define DATA_DIMS MNIST_DIMS
#define OUT_DIM 10
#define LABEL_DIM 1

using namespace Legion;
using namespace std;
using namespace FlexFlow;

struct MoeConfig {
  MoeConfig(void) {
    // MoE layer
    num_exp = 5;
    num_select = 2;
    alpha = 2.0f;
    lambda = 0.04f;
    hidden_size = DATA_DIMS;
    // Encoder layer
    num_attention_heads = 16;
    attention_kdim = attention_vdim = hidden_size / num_attention_heads;
    num_encoder_layers = 6;
  }
  // MoE layer
  int num_exp;
  int num_select;
  float alpha;  // factor overhead tensor size for imbalance
  float lambda; // multiplier for load balance term
  int hidden_size;
  // Encoder layer
  int num_attention_heads;
  int attention_kdim;
  int attention_vdim;
  int num_encoder_layers;
  // Dataset
  std::string dataset_path;
};

class DataLoader {
public:
  DataLoader(FFModel &ff,
             MoeConfig const &moe,
             ParallelTensor input,
             ParallelTensor label);
  static void load_input(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);
  static void load_label(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);
  static void load_entire_dataset(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime);
  void next_batch(FFModel &);
  void reset(void);

public:
  int num_samples, next_index;
  FlexFlow::ParallelTensor full_input, batch_input;
  FlexFlow::ParallelTensor full_label, batch_label;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};
