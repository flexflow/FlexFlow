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
#define MAX_NUM_SAMPLES 65536

using namespace Legion;
using namespace FlexFlow;

struct LLAMAConfig {
  LLAMAConfig(void);
  int hidden_size, n_embd, n_heads, n_layers, sequence_length, vocab_size,
      embedding_prob_drop, layer_norm_epsilon, n_positions, dim, multiple_of,
      norm_eps, attn_pdrop, activation_function, embd_pdrop, resid_pdrop,
      block_size, hidden_dim;
};

class DataLoader {
public:
  DataLoader(FFModel &ff,
             LLAMAConfig const *llamaconfig,
             ParallelTensor const &input,
             ParallelTensor const &pos,
             ParallelTensor const &label);
  void next_batch(FFModel &ff);
  void reset();
  static void load_entire_dataset(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime);
  static void load_input(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);

  static void load_label(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);

  static void load_weights(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);

  static void load_bias(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime); 
  static void load_pos(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);                       
public:
  int num_samples, next_index;
  FlexFlow::ParallelTensor full_input, batch_input;
  FlexFlow::ParallelTensor full_label, batch_label;
  FlexFlow::ParallelTensor full_pos, batch_pos;
  FlexFlow::ParallelTensor ln_f_bias, ln_f_weight;
  std::unordered_map<char const*, float*> weights_pointers;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};