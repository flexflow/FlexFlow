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

#include "llama.h"

namespace FlexFlow {

using namespace Legion;

void LLAMA::create_llama_model(FFModel &ff,
                               InferenceManager &im,
                               Config const &llama_config,
                               int num_pipeline_stages,
                               InferenceMode mode) {
  //------------------------------compute machine views ------------------
  int num_devices = ff.config.workersPerNode * ff.config.numNodes;
  std::vector<MachineView> machine_views;
  for (int i = 0; i < num_devices; i++) {
    MachineView view;
    view.device_type = MachineView::GPU;
    view.ndims = 1;
    view.dim[0] = 1;
    view.stride[0] = 0;
    view.start_device_id = i;
    machine_views.push_back(view);
  }

  std::unordered_map<Tensor, std::vector<MachineView>> mapping;
  std::unordered_map<std::string, Layer *> weights_layers;

  std::cout << "print llama config: " << llama_config.input_path << "-->"
            << llama_config.batchSize << std::endl;

  Tensor input;
  {
    int const token_dims[] = {llama_config.batchSize, llama_config.max_seq_len};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
  }
  mapping[input].push_back(machine_views[0]);

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);
  Tensor token = ff.embedding(input,
                              llama_config.vocab_size,
                              llama_config.dim,
                              AGGR_MODE_NONE,
                              DT_FLOAT,
                              NULL,
                              embed_init);
  Layer *embedding = ff.layers.back();
  weights_layers.emplace("tok_embeddings_weight", embedding);

  int num_transformer_layers_per_stage =
      (32 + num_pipeline_stages - 1) / num_pipeline_stages;

  for (int i = 0; i < 1; i++) {
    // step 1: attention
    std::vector<int> axes = {2};
    Tensor att_norm =
        ff.rms_norm(token, llama_config.norm_eps, llama_config.dim);
    Layer *attention_norm = ff.layers.back();

    if (i % num_transformer_layers_per_stage == 0) {
      // Map att_norm to the next GPU
      // since the size of att_norm is minimum across
      // all tensors
      mapping[att_norm].push_back(
          machine_views[i / num_transformer_layers_per_stage]);
    }

    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_attention_norm_weight",
                           attention_norm);

    // std::cout << "------before att shape";
    // std::cout << att_norm->num_dims << "------\n";
    // for (int i = 0; i < att_norm->num_dims; i++) {
    //   std::cout << att_norm->dims[i] << "------\n";
    // }
    Tensor mha;
    switch (mode) {
      case BEAM_SEARCH_MODE: {
        mha = ff.spec_inc_multihead_self_attention(
            att_norm,
            llama_config.dim,
            llama_config.n_heads,
            llama_config.dim / llama_config.n_heads,
            llama_config.dim / llama_config.n_heads,
            0.0f,
            true,
            false,
            false,
            NULL,
            true);
        break;
      }
      case TREE_VERIFY_MODE: {
        mha = ff.inc_multihead_self_attention_verify(
            att_norm,
            llama_config.dim,
            llama_config.n_heads,
            llama_config.dim / llama_config.n_heads,
            llama_config.dim / llama_config.n_heads,
            0.0f,    /*dropout*/
            true,    /*bias*/
            false,   /*add_bias_kv*/
            false,   /*add_zero_attn*/
            nullptr, /*kernel_initializer*/
            true     /*apply_rotary_embedding*/
        );
        break;
      }
      case INC_DECODING_MODE: {
        mha = ff.inc_multihead_self_attention(
            att_norm,
            llama_config.dim,
            llama_config.n_heads,
            llama_config.dim / llama_config.n_heads,
            llama_config.dim / llama_config.n_heads,
            0.0f,    /*dropout*/
            true,    /*bias*/
            false,   /*add_bias_kv*/
            false,   /*add_zero_attn*/
            nullptr, /*kernel_initializer*/
            true     /*apply_rotary_embedding*/
        );
        break;
      }
      default: {
        assert(false);
      }
    }
    Layer *attention_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_attention_weight",
                           attention_layer);
    token = ff.add(token, mha);

    // step 2: SILU activaion
    Tensor ff_norm =
        ff.rms_norm(token, llama_config.norm_eps, llama_config.dim);
    Layer *ffn_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_ffn_norm_weight",
                           ffn_layer);

    Tensor w1 = ff.dense(ff_norm, llama_config.hidden_dim, AC_MODE_NONE, false);
    Layer *w1_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_feed_forward_w1_weight", w1_layer);

    Tensor w3 = ff.dense(ff_norm, llama_config.hidden_dim, AC_MODE_NONE, false);
    Layer *w3_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_feed_forward_w3_weight", w3_layer);

    Tensor sigmoid = ff.sigmoid(w1);
    Tensor silu = ff.multiply(w1, sigmoid);
    Tensor multi = ff.multiply(silu, w3);

    Tensor w2 = ff.dense(multi, llama_config.dim, AC_MODE_NONE, false);
    Layer *w2_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_feed_forward_w2_weight", w2_layer);
    token = ff.add(token, w2);
  }
  // final normalization and linear
  std::vector<int> axes = {2};
  token = ff.rms_norm(token, 1e-6, 4096);
  Layer *final_norm = ff.layers.back();
  weights_layers.emplace("norm_weight", final_norm);

  Tensor dense = ff.dense(token, llama_config.vocab_size, AC_MODE_NONE, false);
  Layer *final_linear = ff.layers.back();
  weights_layers.emplace("output_weight", final_linear);

  Tensor output;
  if (mode == BEAM_SEARCH_MODE) {
    Tensor softmax = ff.softmax(dense, -1);
    output = ff.beam_top_k(softmax, llama_config.max_beam_width, false);
  } else {
    output = ff.arg_top_k(dense, /*k=*/1, false);
  }

  // Compile the model
  std::cout << "------start compile ----------" << std::endl;
  im.compile_model_and_allocate_buffer(&ff, mapping);
  FileDataLoader fileloader(llama_config.input_path,
                            llama_config.weight_file_path,
                            llama_config.n_heads,
                            llama_config.dim,
                            llama_config.dim / llama_config.n_heads);
  fileloader.load_weights(&ff, weights_layers);
  std::cout << "------load wieght finished----------" << std::endl;

  // init operators
  im.init_operators_inference(&ff);
}

}; // namespace FlexFlow
