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
#include "flexflow/inference.h"

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
  total_sentence = 5;
  total_len = 347;

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
  std::unordered_map<std::string, Layer *> weights_layers;
  Tensor input;
  {
    // 5 inputs
    int const token_dims[] = {ffconfig.batchSize, 1};
    input = ff.create_tensor<2>(token_dims, DT_INT64);
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
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_attention_norm_weight",
                           attention_norm);

    // missing a rotary embedding of q, k, v
    // Tensor q = ff.dense(norm_output, llamaConfig.dim, AC_MODE_RELU, false);
    // Layer *q_layer = ff.layers.back();
    // weights_layers.emplace(
    //     "layers_" + std::to_string(i) + "_attention_wq_weight", q_layer);
    // Tensor k = ff.dense(norm_output, llamaConfig.dim, AC_MODE_RELU, false);
    // Layer *k_layer = ff.layers.back();
    // weights_layers.emplace(
    //     "layers_" + std::to_string(i) + "_attention_wk_weight", k_layer);
    // Tensor v = ff.dense(norm_output, llamaConfig.dim, AC_MODE_RELU, false);
    // Layer *v_layer = ff.layers.back();
    // weights_layers.emplace(
    //     "layers_" + std::to_string(i) + "_attention_wv_weight", v_layer);

    // TODO add a rotary embedding before calling attention
    //    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    Tensor mha = ff.multihead_attention(norm_output,
                                        norm_output,
                                        norm_output,
                                        llamaConfig.dim,
                                        llamaConfig.n_heads,
                                        llamaConfig.dim / llamaConfig.n_heads,
                                        llamaConfig.dim / llamaConfig.n_heads);

    Layer *attention_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_attention_weight", attention_layer);
    token = ff.add(token, mha);

    // step 2: SILU activaion
    Tensor ffn_norm = ff.layer_norm(token, axes, false, llamaConfig.norm_eps);
    Layer *ffn_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_ffn_norm_weight",
                           ffn_layer);
    // hidden dim
    Tensor w1 = ff.dense(ffn_norm, llamaConfig.hidden_dim, AC_MODE_RELU, false);
    Layer *w1_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_feed_forward_w1_weight", w1_layer);

    Tensor w3 = ff.dense(ffn_norm, llamaConfig.hidden_dim, AC_MODE_RELU, false);
    Layer *w3_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_feed_forward_w3_weight", w3_layer);
    Tensor sigmoid = ff.sigmoid(w1);
    Tensor silu = ff.multiply(w1, sigmoid);
    Tensor multi = ff.multiply(silu, w3);
    Tensor w2 = ff.dense(multi, llamaConfig.dim, AC_MODE_RELU, false);
    Layer *w2_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_feed_forward_w2_weight", w2_layer);
    token = ff.add(token, w2);
  }

  std::vector<int> axes = {2};
  token = ff.layer_norm(token, axes, true, llamaConfig.norm_eps);
  Tensor output = ff.dense(token, llamaConfig.vocab_size, AC_MODE_RELU, false);

  // optimizer
  Optimizer *optimizer = new SGDOptimizer(&ff, 0.01f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);

  fprintf(stderr, "----------load inputs--------------");
  // read prompt into input
  ParallelTensor input_pt;
  ff.get_parallel_tensor_from_tensor(input, input_pt);
  DataLoader loader(ff, &llamaConfig, input_pt);

  fprintf(stderr, "----------load weights--------------");
  for (auto &v : weights_layers) {
    std::cout << "weights : " << v.first << "\n";
    Tensor weight = v.second->weights[0];

    if (weight == NULL) {
      std::cout << "no weights : " << v.first << "\n";
      continue;
    }

    size_t volume = 1;
    std::vector<int> dims_vec;
    // std::cout << "shape is: " << std::endl;
    for (int i = 0; i < weight->num_dims; i++) {
      dims_vec.push_back(weight->dims[i]);
      // std::cout << weight->dims[i] << ", ";
      volume *= static_cast<size_t>(weight->dims[i]);
    }

    assert(weight->data_type == DT_FLOAT);
    float *data = (float *)malloc(sizeof(float) * volume);

    if(v.first.find("attention") != std::string::npos){
       std::cout << "attention layer "<< v.first <<std::endl;
       loader.load_attention_weights(data, volume, v.first);
    }else{
       loader.load_from_file(data,
                          volume,
                          "/home/ubuntu/FlexFlow/examples/cpp/LLAMA/weights/" +
                              v.first);
    }
   
    weight->set_tensor<float>(&ff, dims_vec, data);
  }

  ff.init_operators();

  // todo, replace it with inference
  loader.reset();
  ff.reset_metrics();
  

  for(int i = 0; i < 3; i++){
      loader.next_batch(ff);
  }
  // assert(false);

  // // first iteration: total batch/batch size
  // for (int i = 0; i < (llamaConfig.total_sentence / ffconfig.batchSize); i++) {
  //   // second iteration: for each batch, predict one by one token
  //   for (int j = 0; j < llamaConfig.total_len; j++) {
  //     // input shape, batch_size * 1 = 5 * 1
  //     std::cout << "iteration" << j << ", ";
  //     ff.forward();
  //     loader.next_batch(ff);
      
  //   }
  //   loader.reset();
  //   // todo process one sentence
  // }

  fprintf(stderr, "----------inference end--------------");
}