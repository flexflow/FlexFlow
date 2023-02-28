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

#include "data_generator.h"
#include "flexflow/model.h"
#define MAX_NUM_SAMPLES 1000
#define MNIST_DIMS 28 * 28
#define DATA_DIM MNIST_DIMS

using namespace Legion;
using namespace std;
using namespace FlexFlow;

struct MoeConfig {
  MoeConfig(void) {
    //----------------------- Input/output data ------------------------
    token_dim = DATA_DIM;
    sequence_length = 10;
    batch_size = 32;
    out_dim = 15;
    num_labels = out_dim;
    num_layers = 12;
    //----------------------- Inference parameters ---------------------
    // total number of requests processed as part of the simulation
    total_requests = 256;
    poisson_distribution = true;
    // average number of request arrivals per second
    arrival_rate = 25;
    num_inflight_batches = 10;
    //----------------------- MoE layer --------------------------------
    // total number of experts
    num_exp = 128;
    // number of experts in each block of fused experts
    experts_per_block = 32;
    // number of experts to route each token to
    num_select = 2;
    // expert capacity parameters
    alpha = 2.0f;   // factor overhead tensor size for imbalance
    lambda = 0.04f; // multiplier for load balance term
    // expert hidden size
    hidden_size = DATA_DIM;
    //----------------------- Rest of model parameters ------------------
    // Encoder layer
    num_attention_heads = 16;
    attention_kdim = attention_vdim = hidden_size / num_attention_heads;
    num_encoder_layers = 1;
  }

  // Input/output data
  int token_dim;
  int sequence_length;
  int batch_size;
  int out_dim;
  int num_labels;
  int num_layers;
  std::string dataset_path;
  // Inference parameters
  int total_requests;
  bool poisson_distribution;
  double arrival_rate;
  int num_inflight_batches;
  // MoE layer
  int num_exp;
  int experts_per_block;
  int num_select;
  float alpha;
  float lambda;
  int hidden_size;
  // Model parameters
  int num_attention_heads;
  int attention_kdim;
  int attention_vdim;
  int num_encoder_layers;
};

class DataLoader {
public:
  DataLoader(FFModel &ff,
             MoeConfig const &moeConfig,
             DataGenerator &data_generator,
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
  void next_batch(FFModel &, size_t);
  void reset(void);

public:
  int num_samples, next_index;
  FlexFlow::ParallelTensor full_input, batch_input;
  FlexFlow::ParallelTensor full_label, batch_label;
  struct DataLoaderInput {
    MoeConfig const &_moeConfig;
    DataGenerator &_data_generator;
  };
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};
