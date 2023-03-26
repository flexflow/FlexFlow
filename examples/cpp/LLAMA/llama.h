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
    // std::cout << "size seee" << std::endl;
    // std::cout << loaded_data_size << std::endl;
    // std::cout << in_get_size << std::endl;
    if (in_get_size != loaded_data_size) {
      std::cout << "load data error";
      return;
    }

    // std::cout << "finish loading input";
    assert(size == host_array.size());

    // normal
    long data_index = 0;
    for (auto v : host_array) {
      ptr[data_index++] = v;
    }
    in.close();
  }

  template <typename T>
  static void
      load_attention_weights(T *ptr, size_t size, std::string layer_name) {

    // get files
    // layers_3_attention_weight
    std::string q_file = "/home/ubuntu/FlexFlow/examples/cpp/LLAMA/weights/" +
                         layer_name.substr(0, layer_name.find("attention")) +
                         "attention_wq_weight";
    std::string k_file = "/home/ubuntu/FlexFlow/examples/cpp/LLAMA/weights/" +
                         layer_name.substr(0, layer_name.find("attention")) +
                         "attention_wk_weight";
    std::string v_file = "/home/ubuntu/FlexFlow/examples/cpp/LLAMA/weights/" +
                         layer_name.substr(0, layer_name.find("attention")) +
                         "attention_wk_weight";
    std::string o_file = "/home/ubuntu/FlexFlow/examples/cpp/LLAMA/weights/" +
                         layer_name.substr(0, layer_name.find("attention")) +
                         "attention_wv_weight";
    std::vector<std::string> weight_files = {q_file, k_file, v_file, o_file};

    size_t index = 0;

    for (auto file : weight_files) {
      size_t partial_size = size / 4;
      std::ifstream in(file, std::ios::in | std::ios::binary);
      std::vector<T> host_array(partial_size);
      size_t loaded_data_size = sizeof(T) * partial_size;
      in.seekg(0, in.end);
      in.seekg(0, in.beg);
      in.read((char *)host_array.data(), loaded_data_size);
      size_t in_get_size = in.gcount();

      if (in_get_size != loaded_data_size) {
        std::cout << "load data error";
        return;
      }
      assert(partial_size == host_array.size());

      size_t offset = index * 4096 * 4096;
      size_t one_head_size = 4096 * 128;
      size_t data_index = 0;

      for (size_t i = 0; i < one_head_size; i++) {
        for (size_t j = 0; j < 32; j++) {
          ptr[j * one_head_size + i + index] =
              host_array.at(data_index++);
        }
      }

      in.close();
      index++;
    }
  }

public:
  int num_samples, next_index, next_token_idx, next_batch_index;
  FlexFlow::ParallelTensor full_input, batch_input;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
  int token_idx;
  int batch_idx;
};