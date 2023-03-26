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
  LLAMAConfig::LLAMAConfig(void) {
    // todo read from config/param file
    n_layers = 32;
    vocab_size = 32000;
    n_heads = 32;
    dim = 4096;
    multiple_of = 256;
    norm_eps = 1e-6;
    total_sentence = 5;
    sentence_len = 347;
    batchSize = 5;

    // todo from args
    weight_file_path = "/home/ubuntu/FlexFlow/examples/cpp/LLAMA/weights/";
    input_path = "";

    // hidden dim
    hidden_dim = 4 * dim;
    hidden_dim = int(2 * hidden_dim / 3);
    hidden_dim =
        multiple_of * int((hidden_dim + multiple_of - 1) / multiple_of);
  }
  int n_heads, n_layers, vocab_size, dim, multiple_of, norm_eps, hidden_dim,
      total_sentence, sentence_len batchSize;
  std::string weight_file_path;
  std::string input_path;
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
  static void load_from_file(T *ptr, size_t size, std::string filename);

  template <typename T>
  static void load_attention_weights(T *ptr,
                                     size_t size,
                                     std::string layer_name,
                                     std::string weight_path);

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