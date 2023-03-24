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
  embedding_prob_drop = 0.1;
  n_embd = 768;
  resid_pdrop = 0.1;
  vocab_size = 32000;
  block_size = 1024;
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
  std::vector<Layer *> weights_layers;
  FFModel ff(ffconfig);
  // load weights

  // TODO init params from pre-trained model
  Tensor input;
  Tensor pos;
  {
    // 5 inputs
    int const token_dims[] = {ffconfig.batchSize, 1};
    int const pos_dims[] = {ffconfig.batchSize, 1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
  }
  // word&position embedding
  std::cout << "------input shaoe";                          
  std::cout << input->num_dims << "------\n";
  for(int i = 0; i < input->num_dims; i++){
    std::cout << input->dims[i] << "------\n";
  }      

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);
  Tensor token = ff.embedding(input,
                              llamaConfig.vocab_size,
                              llamaConfig.dim,
                              AGGR_MODE_NONE,
                              DT_FLOAT,
                              NULL,
                              embed_init);

  std::cout << "------token shape";                          
  std::cout << token->num_dims << "------\n";
  for(int i = 0; i < token->num_dims; i++){
    std::cout << token->dims[i] << "------\n";
  }                      

  // n transformer blocks impl
  for (int i = 0; i < 4; i++) {
    // step 1: normalization on the embedding dim
    std::vector<int> axes = {2};
    Tensor norm_output =
        ff.layer_norm(token, axes, false, llamaConfig.norm_eps);

    std::cout << "------get q, k, v" <<"\n";  
    std::cout << "-----norm shape";                          
    std::cout << norm_output->num_dims << "------\n";
    for(int i = 0; i < norm_output->num_dims; i++){
      std::cout << norm_output->dims[i] << "------\n";
    }    
    // missing a rotary embedding of q, k, v
    Tensor q = ff.dense(norm_output, llamaConfig.dim, AC_MODE_RELU, false);
    Tensor k = ff.dense(norm_output, llamaConfig.dim, AC_MODE_RELU, false);
    Tensor v = ff.dense(norm_output, llamaConfig.dim, AC_MODE_RELU, false);

    std::vector<int> dims = {ffconfig.batchSize, 1, llamaConfig.n_heads, llamaConfig.dim / llamaConfig.n_heads};
    Tensor xq = ff.reshape(q, dims);
    Tensor xk = ff.reshape(k ,dims);
    Tensor xv = ff.reshape(v, dims);

    std::cout << "------attention" <<"\n";   
    std::cout << llamaConfig.hidden_dim <<"\n";

    // TODO add a rotary embedding before calling attention
    //    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    Tensor mha = ff.multihead_attention(xq,
                                        xk,
                                        xv,
                                        llamaConfig.dim,
                                        llamaConfig.n_heads,
                                        llamaConfig.dim,
                                        llamaConfig.dim);
    token = ff.add(token, mha);

    // step 2: SILU activaion
    Tensor ffn_norm = ff.layer_norm(token, axes, false, llamaConfig.norm_eps);
    // hidden dim
    Tensor w1 = ff.dense(ffn_norm, llamaConfig.hidden_dim, AC_MODE_RELU, false);
    Tensor w3 = ff.dense(ffn_norm, llamaConfig.hidden_dim, AC_MODE_RELU, false);
    Tensor sigmoid = ff.sigmoid(w1);
    Tensor silu = ff.multiply(w1, sigmoid);
    Tensor multi = ff.multiply(silu, w3);
    Tensor w2 = ff.dense(multi, llamaConfig.dim, AC_MODE_RELU, false);
    token = ff.add(token, w2);
  }
  
  std::cout << "final norm" <<"\n";   
  std::vector<int> axes = {2};
  token = ff.layer_norm(token, axes, true, llamaConfig.norm_eps);
  std::cout << "final dense" <<"\n";   
  Tensor output = ff.dense(token, llamaConfig.vocab_size, AC_MODE_RELU, false);

  // optimizer
  Optimizer *optimizer = new SGDOptimizer(&ff, 0.01f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);

  fprintf(stderr, "----------compile end--------------");
  // read data into input
  // ParallelTensor input_pt, label_pt, pos_pt;
  // ff.get_parallel_tensor_from_tensor(input, input_pt);
  // ff.get_parallel_tensor_from_tensor(pos, pos_pt);
  // ff.get_parallel_tensor_from_tensor(ff.label_tensor, label_pt);
  // DataLoader loader(ff, &minigptconfig, input_pt, pos_pt, label_pt);

  // // set weight tensors
  // ParallelTensor weights_pt, bias_pt;
  // // copy data to weights tensor
  // Layer *l = weights_layers[0];
  // Tensor weight = l->weights[0];
  // // ff.get_parallel_tensor_from_tensor(l->weights[0], weights_pt);

  // // load weight data
  // int size = minigptconfig.n_embd * 3 * ffconfig.batchSize;
  // float data[size];
  // for (int i = 0; i < size; i++) {
  //   data[i] = ((float)std::rand()) / RAND_MAX;
  // }
  // std::vector<int> dims_vec;
  // for (int i = 0; i < weight->num_dims; i++) {
  //   dims_vec.push_back(weight->dims[i]);
  // }
  // weight->set_tensor<float>(&ff, dims_vec, data);

  // loader.next_batch(ff);
  // loader.reset();
  // ff.init_operators();

  // // train
  // for (int epoch = 0; epoch < ffconfig.epochs; epoch++) {
  //   // loader.reset();
  //   ff.reset_metrics();
  //   int iterations = loader.num_samples / ffconfig.batchSize;
  //   for (int iter = 0; iter < iterations; iter++) {
  //     // // Only load data once for random input
  //     if (iter == 0 && epoch == 0) {
  //       loader.next_batch(ff);
  //     }
  //     runtime->begin_trace(ctx, 111 /*trace_id*/);
  //     ff.forward();
  //     // decode
  //     //  ff.zero_gradients();
  //     //  ff.backward();
  //     //  ff.update();
  //     runtime->end_trace(ctx, 111 /*trace_id*/);
  //   }
  // }

  // fprintf(stderr, "----------train end--------------");
}