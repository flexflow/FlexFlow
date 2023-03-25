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

using namespace Legion;

LegionRuntime::Logger::Category log_app("minigpt");

// future read from config file
LLAMAConfig::LLAMAConfig(void) {
  // todo read from config/param file
  n_layers = 32;
  vocab_size = 32000;
  n_heads = 32;
  dim = 4096;
  multiple_of = 256;
  norm_eps = 1e-6;

  // hidden dim
  hidden_dim = 4 * dim;
  hidden_dim = int(2 * hidden_dim / 3);
  hidden_dim = multiple_of * int((hidden_dim + multiple_of - 1) / multiple_of);
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  LLAMAConfig llamaConfig;
  FFModel ff(ffconfig);

  // load the weight from binary file
  std::unordered_map<char const *, Layer *> weights_layers;
  Tensor input;
  {
    // 5 inputs
    int const token_dims[] = {ffconfig.batchSize, 1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
  }
  // word&position embedding
  // std::cout << "------input shaoe";
  // std::cout << input->num_dims << "------\n";
  // for(int i = 0; i < input->num_dims; i++){
  //   std::cout << input->dims[i] << "------\n";
  // }

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);
  Tensor token = ff.embedding(input,
                              llamaConfig.vocab_size,
                              llamaConfig.dim,
                              AGGR_MODE_NONE,
                              DT_FLOAT,
                              NULL,
                              embed_init);
  Layer *embedding = ff.layers.back();
  weights_layers.emplace("tok_embeddings_weight", embedding);

  // std::cout << "------token shape";
  // std::cout << token->num_dims << "------\n";
  // for(int i = 0; i < token->num_dims; i++){
  //   std::cout << token->dims[i] << "------\n";
  // }

  // n transformer blocks impl
  for (int i = 0; i < 4; i++) {
    // step 1: normalization on the embedding dim
    std::vector<int> axes = {2};
    Tensor norm_output =
        ff.layer_norm(token, axes, false, llamaConfig.norm_eps);
    Layer *attention_norm = ff.layers.back();
    weights_layers.emplace("layers_" + i + "_attention_norm_weight",
                           attention_norm);

    // missing a rotary embedding of q, k, v
    Tensor q = ff.dense(norm_output, llamaConfig.dim, AC_MODE_RELU, false);
    Layer *q_layer = ff.layers.back();
    weights_layers.emplace("layers_" + i + "_attention_wq_weight", q_layer);
    Tensor k = ff.dense(norm_output, llamaConfig.dim, AC_MODE_RELU, false);
    Layer *k_layer = ff.layers.back();
    weights_layers.emplace("layers_" + i + "_attention_wk_weight", k_layer);
    Tensor v = ff.dense(norm_output, llamaConfig.dim, AC_MODE_RELU, false);
    Layer *v_layer = ff.layers.back();
    weights_layers.emplace("layers_" + i + "_attention_wv_weight", v_layer);

    // TODO add a rotary embedding before calling attention
    //    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    Tensor mha = ff.multihead_attention(q,
                                        k,
                                        v,
                                        llamaConfig.dim,
                                        llamaConfig.n_heads,
                                        llamaConfig.dim / llamaConfig.n_heads,
                                        llamaConfig.dim / llamaConfig.n_heads);

    // std::cout << "-----mha shape";
    // std::cout << mha->num_dims << "------\n";
    // for(int i = 0; i < mha->num_dims; i++){
    //   std::cout << mha->dims[i] << "------\n";
    // }
    // dims = {ffconfig.batchSize, 1, llamaConfig.dim};
    // mha = ff.reshape(mha, dims);
    token = ff.add(token, mha);

    // step 2: SILU activaion
    Tensor ffn_norm = ff.layer_norm(token, axes, false, llamaConfig.norm_eps);
    Layer *ffn_layer = ff.layers.back();
    weights_layers.emplace("layers_" + i + "_ffn_norm_weight", ffn_layer);
    // hidden dim
    Tensor w1 = ff.dense(ffn_norm, llamaConfig.hidden_dim, AC_MODE_RELU, false);
    Layer *w1_layer = ff.layers.back();
    weights_layers.emplace("layers_" + i + "_feed_forward_w1_weight", w1_layer);

    Tensor w3 = ff.dense(ffn_norm, llamaConfig.hidden_dim, AC_MODE_RELU, false);
    Layer *w3_layer = ff.layers.back();
    weights_layers.emplace("layers_" + i + "_feed_forward_w3_weight", w3_layer);
    Tensor sigmoid = ff.sigmoid(w1);
    Tensor silu = ff.multiply(w1, sigmoid);
    Tensor multi = ff.multiply(silu, w3);
    Tensor w2 = ff.dense(multi, llamaConfig.dim, AC_MODE_RELU, false);
    Layer *w2_layer = ff.layers.back();
    weights_layers.emplace("layers_" + i + "_feed_forward_w1_weight", w2_layer);
    token = ff.add(token, w2);
  }

  std::cout << "final norm"
            << "\n";
  std::vector<int> axes = {2};
  token = ff.layer_norm(token, axes, true, llamaConfig.norm_eps);
  std::cout << "final dense"
            << "\n";
  Tensor output = ff.dense(token, llamaConfig.vocab_size, AC_MODE_RELU, false);

  // optimizer
  Optimizer *optimizer = new SGDOptimizer(&ff, 0.01f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);

  fprintf(stderr, "----------loading weights--------------");

  for (auto &v : weights_layers) {
    if (weights_pointers.find(v.first) != weights_pointers.end()) {
      Tensor weight = v.second->weights[0];
      assert(weight != NULL) weight->set_tensor<float>(
          &ff, weight.dims, weights_pointers.find(v.first));
    }
  }

  read prompt into input
  ParallelTensor input_pt;
  ff.get_parallel_tensor_from_tensor(input, input_pt);
  DataLoader loader(ff, &llamaConfig, input_pt);

  loader.next_batch(ff);
  loader.reset();
  ff.init_operators();

  // fprintf(stderr, "----------inference end--------------");
}