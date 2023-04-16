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

LegionRuntime::Logger::Category log_app("llama");

void parse_input_args(char **argv, int argc, LLAMAConfig &config) {
  for (int i = 1; i < argc; i++) {

    // input
    if (!strcmp(argv[i], "--dataset")) {
      config.input_path = std::string(argv[++i]);
      continue;
    }

    // weights
    if (!strcmp(argv[i], "--weights")) {
      config.weight_file_path = std::string(argv[++i]);
      continue;
    }
  }
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  LLAMAConfig llamaConfig;
  FFModel ff(ffconfig);
  std::unordered_map<std::string, Layer *> weights_layers;

  // InputArgs const &command_args = HighLevelRuntime::get_input_args();
  // char **argv = command_args.argv;
  // int argc = command_args.argc;
  // parse_input_args(argv, argc, llamaConfig);

  std::cout << "print llama config: " << llamaConfig.input_path << "-->"
            << llamaConfig.batchSize;

  //------------------------------ build the model --------------------------
  Tensor input;
  {
    int const token_dims[] = {llamaConfig.batchSize, llamaConfig.max_seq_len};
    input = ff.create_tensor<2>(token_dims, DT_INT64);
  }

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
  // for (int i = 0; i < token->num_dims; i++) {
  //   std::cout << token->dims[i] << "------\n";
  // }

  // n transformer blocks impl
  for (int i = 0; i < 1; i++) {
    // step 1: attention
    std::vector<int> axes = {2};
    Tensor att_norm = ff.rms_norm(token, llamaConfig.norm_eps, llamaConfig.dim);
    Layer *attention_norm = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_attention_norm_weight",
                           attention_norm);
    Tensor mha =
        ff.inc_multihead_self_attention(att_norm,
                                        llamaConfig.dim,
                                        llamaConfig.n_heads,
                                        llamaConfig.dim / llamaConfig.n_heads,
                                        llamaConfig.dim / llamaConfig.n_heads,
                                        0.0f,
                                        true,
                                        false,
                                        false,
                                        NULL,
                                        true);
    Layer *attention_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_attention_weight",
                           attention_layer);
    token = ff.add(token, mha);

    // step 2: SILU activaion
    Tensor ff_norm = ff.rms_norm(token, llamaConfig.norm_eps, llamaConfig.dim);
    Layer *ffn_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_ffn_norm_weight",
                           ffn_layer);

    Tensor w1 = ff.dense(ff_norm, llamaConfig.hidden_dim, AC_MODE_NONE, false);
    Layer *w1_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_feed_forward_w1_weight", w1_layer);

    Tensor w3 = ff.dense(ff_norm, llamaConfig.hidden_dim, AC_MODE_NONE, false);
    Layer *w3_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_feed_forward_w3_weight", w3_layer);

    Tensor sigmoid = ff.sigmoid(w1);
    Tensor silu = ff.multiply(w1, sigmoid);
    Tensor multi = ff.multiply(silu, w3);

    Tensor w2 = ff.dense(multi, llamaConfig.dim, AC_MODE_NONE, false);
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
  Tensor dense = ff.dense(token, llamaConfig.vocab_size, AC_MODE_NONE, false);
  Layer *final_linear = ff.layers.back();
  weights_layers.emplace("output_weight", final_linear);
  Tensor output = ff.arg_top_k(dense, /*k=*/1, false);

  //------------------- compile the model --------------------------------
  std::cout << "------start compile ----------" << std::endl;
  InferenceManager im(&ff, llamaConfig.batchSize, 1);
  im.compile_model_and_allocate_buffer();

  std::cout << "------init ops----------" << std::endl;
  im.init_operators_inference();
  std::cout << "------model compiled and init ----------" << std::endl;

  //------------------------------ load inputs --------------------------
  std::cout << "------create dataloaders ----------" << std::endl;
  // read prompt into input
  ParallelTensor input_pt;
  ff.get_parallel_tensor_from_tensor(input, input_pt);
  assert(im.tensor_buffer.find(input_pt) != im.tensor_buffer.end());
  std::cout << im.tensor_buffer[input_pt].size() << std::endl;
  DataLoader loader(ff, &llamaConfig, im.tensor_buffer[input_pt].at(0));

  //------------------------------ load weights---------------------------
  for (auto &v : weights_layers) {
    Tensor weight = v.second->weights[0];
    std::cout << "weights layer: " << v.first << "\n";

    if (weight == NULL) {
      std::cout << "op no weights : " << v.first << "\n";
      continue;
    }

    size_t volume = 1;
    std::vector<int> dims_vec;
    for (int i = 0; i < weight->num_dims; i++) {
      dims_vec.push_back(weight->dims[i]);
      volume *= weight->dims[i];
    }

    assert(weight->data_type == DT_FLOAT);
    float *data = (float *)malloc(sizeof(float) * volume);

    if (v.first.find("attention_w") != std::string::npos) {
      loader.load_attention_weights(
          data, volume, v.first, llamaConfig.weight_file_path);

    } else {
      loader.load_from_file(
          data, volume, llamaConfig.weight_file_path + v.first);
      if (v.first.find("attention_norm") != std::string::npos) {
        // std::cout << "norm weight data" << std::endl;
        // for (int i = 0; i < 100; i++) {
        //   std::cout << data[i] << ", ";
        // }
      }
    }

    ParallelTensor weight_pt;
    ff.get_parallel_tensor_from_tensor(weight, weight_pt);
    weight_pt->set_tensor<float>(&ff, dims_vec, data);
  }
  std::cout << "------load wieght finished----------" << std::endl;

  //------------------------------ do inference---------------------------
  int processed_requests = 0;
  std::map<int, Future> future_handlers;
  std::map<int, BatchConfig *> batch_configs;
  BatchConfig *bc = nullptr;
  std::map<size_t, long> batch_predictions[1];
  loader.reset();

  bool new_req = true;

  while (processed_requests < llamaConfig.sentence_len) {
    int bid = 0;
    size_t max_reqs, max_tkns;
    if (future_handlers.find(bid) == future_handlers.end()) {
      bc = new BatchConfig();
    } else {
      // have luanched this bid
      Future future = future_handlers[bid];
      if (!future.is_ready(true /*subscribe*/)) {
        continue;
      } else {
        std::cout << "future is ready...." << std::endl;
      }
      // process end
      InferenceResult ir = future.get_result<InferenceResult>();
      bc = batch_configs[bid];

      std::cout << "store outputs start...." << std::endl;
      loader.store_outputs(bc, ir, batch_predictions[bid]);
      processed_requests += bc->update_results(ir);

      if (!new_req) {
        break;
      }
      new_req = false;
    }
    // batch cofig register 5 reqs
    // init length relate to the min_prompt_size for llama
    if (new_req) {
      for (int i = 0; i < llamaConfig.batchSize; i++) {
        assert(bc->register_new_request(i, llamaConfig.max_seq_len, 347));
      }
    }

    bc->prepare_next_batch();
    std::cout << "new tokens: " << bc->num_active_tokens();
    loader.next_batch(ff, bc, batch_predictions[bid]);

    FutureMap fm = im.inference(bid, *bc);
    assert(fm.get_future_map_domain().get_volume() == 1);
    future_handlers[bid] = fm.get_future(0);
    batch_configs[bid] = bc;
  }

  // float* data
  std::cout << "----------inference finished--------------" << std::endl;
}
