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
#define MAX_TOKEN_LEN 32000

using namespace Legion;
using namespace FlexFlow;

struct LLAMAConfig {
  LLAMAConfig(void);
  int hidden_size, n_embd, n_heads, n_layers, sequence_length, vocab_size,
      embedding_prob_drop, layer_norm_epsilon, n_positions, dim, multiple_of,
      norm_eps, attn_pdrop, activation_function, embd_pdrop, resid_pdrop,
      block_size, hidden_dim, total_sentence, total_len;
};

class DataLoader {
public:
  DataLoader(FFModel &ff,
             LLAMAConfig const *llamaconfig,
             ParallelTensor const &input);
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

  template <typename T>
  static void load_from_file(T *ptr, size_t size, std::string filename) {

    // std::cout << "start loading input";
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    std::vector<T> host_array(size);
    size_t loaded_data_size = sizeof(T) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
    in.read((char *)host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    std::cout << "size seee" << std::endl;
    std::cout << loaded_data_size << std::endl;
    std::cout << in_get_size << std::endl;
    if (in_get_size != loaded_data_size) {
      std::cout << "load data error";
      return;
    }

    std::cout << "finish loading input";
    assert(size == host_array.size());
    long index = 0;
    for (auto i = host_array.begin(); i != host_array.end(); i++) {
      ptr[index++] = *i;
    }
    // ptr = (T*)host_array.data();
    in.close();
  }

  template <typename T>
  static void load_attention_weights(T *ptr, size_t size, std::string layer_name) {

    // std::cout << "start loading input";
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    std::vector<T> host_array(size);
    size_t loaded_data_size = sizeof(T) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
    in.read((char *)host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    std::cout << "size seee" << std::endl;
    std::cout << loaded_data_size << std::endl;
    std::cout << in_get_size << std::endl;
    if (in_get_size != loaded_data_size) {
      std::cout << "load data error";
      return;
    }

    std::cout << "finish loading input";
    assert(size == host_array.size());
    long index = 0;
    for (auto i = host_array.begin(); i != host_array.end(); i++) {
      ptr[index++] = *i;
    }
    // ptr = (T*)host_array.data();
    in.close();
  }

public:
  int num_samples, next_index, next_token_idx;
  FlexFlow::ParallelTensor full_input, batch_input;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
  int token_idx[MAX_NUM_SAMPLES];
};