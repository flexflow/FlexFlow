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

#include "minigpt.h"

using namespace Legion;

LegionRuntime::Logger::Category log_app("minigpt");

// future read from config file
MiniGPTConfig::MiniGPTConfig(void) {
  // todo read from config/param file
  n_layer = 6;
  embedding_prob_drop = 0.1;
  n_embd = 768;
  resid_pdrop = 0.1;
  vocab_size = 50257;
  block_size = 1024;
  n_head = 12;
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  MiniGPTConfig minigptconfig;
  std::vector<Layer*> weights_layers;
  FFModel ff(ffconfig);


  // todo init params from pre-trained model
  Tensor input;
  Tensor pos;
  {
    int const token_dims[] = {ffconfig.batchSize, 10, minigptconfig.n_embd};
    int const pos_dims[] = {ffconfig.batchSize, 10, minigptconfig.n_embd};
    input = ff.create_tensor<3>(token_dims, DT_FLOAT);
    pos = ff.create_tensor<3>(pos_dims, DT_FLOAT);
  }

  // word&position embedding
  // Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);
  // Tensor token_embedding = ff.embedding(input,
  //                                       minigptconfig.vocab_size,
  //                                       minigptconfig.n_embd,
  //                                       AGGR_MODE_SUM,
  //                                       DT_FLOAT,
  //                                       NULL,
  //                                       embed_init);
  // Tensor position_embedding = ff.embedding(pos,
  //                                          minigptconfig.block_size,
  //                                          minigptconfig.n_embd,
  //                                          AGGR_MODE_SUM,
  //                                          DT_FLOAT,
  //                                          NULL,
  //                                          embed_init);
  Tensor x = ff.add(input, pos);
  x = ff.dropout(x, minigptconfig.embedding_prob_drop);

  // n-layers transformer block
  for (int i = 0; i < minigptconfig.n_layer; i++) {
    // get q, k, v
    // float const *data = NULL;
    std::vector<int> axes = {2};
    x = ff.layer_norm(x, axes, true, 1e-5);
    // //get the latest layer
   

    Tensor sp_1 = ff.dense(x, minigptconfig.n_embd * 3, AC_MODE_RELU, false);
    Layer *l = ff.layers.back();
    weights_layers.push_back(l);
    //get Tensor access
    // weight->set_tensor(ff, 0, data);
    // bias->set_tensor(ff, 0, data);
    Tensor *splited_tensor = new Tensor[3];

 

    std::vector<int> split = {
        minigptconfig.n_embd, minigptconfig.n_embd, minigptconfig.n_embd};
    ff.split(sp_1, splited_tensor, split, 2);
    Tensor q = splited_tensor[0];
    Tensor k = splited_tensor[1];
    Tensor v = splited_tensor[2];
    assert(q != NULL);
    assert(k != NULL);
    assert(v != NULL);
    // multihead attention
    Tensor mha = ff.multihead_attention(q,
                                        k,
                                        v,
                                        minigptconfig.n_embd,
                                        minigptconfig.n_head,
                                        minigptconfig.n_embd,
                                        minigptconfig.n_embd);
    Tensor ln_1 = ff.dense(mha, minigptconfig.n_embd, AC_MODE_RELU, false);                                  


    x = ff.add(x, ln_1);
    // mlp
    Tensor c_fc = ff.dense(x, minigptconfig.n_embd * 4, AC_MODE_RELU, false);
    Tensor act = ff.gelu(c_fc);
    Tensor c_proj = ff.dense(act, minigptconfig.n_embd, AC_MODE_RELU, false);
    Tensor dropout = ff.dropout(c_proj, minigptconfig.resid_pdrop);
    x = ff.add(x, dropout);
  }

  std::vector<int> axes = {2};
  x = ff.layer_norm(x, axes, true, 1e-5);
  // fprintf(stderr, "--final dim---");
  //    for(int i = 0; i < ff.label_tensor->num_dims; i++){
  //       fprintf(stderr, std::to_string(ff.label_tensor->dims[i]).c_str());
  //       fprintf(stderr, "-----");
  //   }
  // x = ff.dense(x, minigptconfig.vocab_size);

  // optimizer
  Optimizer *optimizer = new SGDOptimizer(&ff, 0.01f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);

  fprintf(stderr, "----------compile end--------------");
  // read data into input
  ParallelTensor input_pt, label_pt, pos_pt;
  ff.get_parallel_tensor_from_tensor(input, input_pt);
  ff.get_parallel_tensor_from_tensor(pos, pos_pt);
  ff.get_parallel_tensor_from_tensor(ff.label_tensor, label_pt);
  DataLoader loader(ff, &minigptconfig, input_pt, pos_pt, label_pt);

  //set weight tensors
  ParallelTensor weights_pt, bias_pt;
  //copy data to weights tensor
  Layer *l = weights_layers[0];
  Tensor weight = l->weights[0];
  // ff.get_parallel_tensor_from_tensor(l->weights[0], weights_pt);

  //mock weight data
  int size = minigptconfig.n_embd * 3 * ffconfig.batchSize;
  float data[size];
  for(int i = 0; i < size; i++){
    data[i] = ((float)std::rand()) / RAND_MAX;
  }
  std::vector<int> dims_vec;
  for (int i = 0; i <weight->num_dims; i++) {
    dims_vec.push_back(weight->dims[i]);
  }
  weight->set_tensor<float>(&ff, dims_vec, data);
  



  loader.next_batch(ff);
  loader.reset();
  ff.init_operators();

  //train
  for (int epoch = 0; epoch < ffconfig.epochs; epoch++) {
    // loader.reset();
    ff.reset_metrics();
    int iterations = loader.num_samples / ffconfig.batchSize;
    for (int iter = 0; iter < iterations; iter++) {
      // // Only load data once for random input
      if (iter == 0 && epoch == 0) {
        loader.next_batch(ff);
      }
      runtime->begin_trace(ctx, 111 /*trace_id*/);
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
      runtime->end_trace(ctx, 111 /*trace_id*/);
    }
  }

  fprintf(stderr, "----------train end--------------");
}