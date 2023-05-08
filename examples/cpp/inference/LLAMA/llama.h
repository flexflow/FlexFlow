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

#include "flexflow/batch_config.h"
#include "flexflow/inference.h"
#include "flexflow/model.h"
#define MAX_NUM_SAMPLES 65536
#define MAX_TOKEN_LEN 32000

using namespace Legion;
using namespace FlexFlow;

struct LLAMAConfig {
  LLAMAConfig(void) {
    // todo read from config/param file
    n_layers = 32;
    vocab_size = 32000;
    n_heads = 32;
    dim = 4096;
    multiple_of = 256;
    norm_eps = 1e-6;
    total_sentence = 5;
    sentence_len = 347;
    max_gen_length = 256;
    batchSize = 5;
    total_requests = 2560;
    incremental_mode = true;
    sequence_length = BatchConfig::MAX_SEQ_LENGTH;
    max_seq_len = 8;

    // todo from args
    weight_file_path =
        "/home/ubuntu/FlexFlow_Inference/examples/cpp/inference/LLAMA/weights/";
    input_path = "/home/ubuntu/FlexFlow/examples/cpp/inference/LLAMA/tokens/"
                 "llama_demo_tokens";

    // hidden dim
    hidden_dim = 4 * dim;
    hidden_dim = int(2 * hidden_dim / 3);
    hidden_dim =
        multiple_of * int((hidden_dim + multiple_of - 1) / multiple_of);
  }
  int n_heads, n_layers, vocab_size, dim, multiple_of, hidden_dim,
      total_sentence, sentence_len, batchSize, total_requests, incremental_mode,
      sequence_length, max_gen_length, max_seq_len;
  float norm_eps;
  std::string weight_file_path;
  std::string input_path;
};

class DataLoader {
public:
  DataLoader(FFModel &ff,
             LLAMAConfig const *llamaconfig,
             ParallelTensor const &input);
  void next_batch(FFModel &ff,
                  BatchConfig *bc,
                  std::map<size_t, long> &batch_predictions);
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
  static void load_attention_weights(T *dst_ptr,
                                     size_t total_weights_size,
                                     int num_heads,
                                     size_t hidden_dim,
                                     size_t qkv_inner_dim,
                                     std::string layer_name,
                                     std::string weight_path);
  void store_outputs(BatchConfig *bc,
                     InferenceResult const &ir,
                     std::map<size_t, long> &batch_predictions);

public:
  int num_samples, next_index, next_token_idx, next_batch_index;
  std::map<size_t, std::vector<int>> outputs;
  FlexFlow::ParallelTensor full_input, batch_input;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
  int token_idx;
  int batch_idx;
};

struct DataLoaderNextBatchInput {
  // BatchConfig::SampleIdxs const &meta;
  BatchConfig *bc;
  std::map<size_t, long> const &prev_batch_preds;
};
