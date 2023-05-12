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

#include "opt.h"
#include "flexflow/inference.h"
#include <cmath>

using namespace Legion;

LegionRuntime::Logger::Category log_app("opt");

void parse_input_args(char **argv, int argc, OptConfig &config) {
  for (int i = 1; i < argc; i++) {
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
  OptConfig optConfig;
  FFModel ff(ffconfig);

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv, argc, optConfig);

  //------------------------------compute machine views ------------------
  int num_devices = ffconfig.workersPerNode * ffconfig.numNodes;
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

  //------------------------------ build the model --------------------------
  Tensor input;
  Tensor position_input;
  {
    int const token_dims[] = {BatchConfig::MAX_NUM_TOKENS, 1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
    position_input = ff.create_tensor<2>(token_dims, DT_INT32);
  }

  mapping[input].push_back(machine_views[0]);
  mapping[position_input].push_back(machine_views[0]);

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);
  std::vector<int> axes = {0};

  Tensor token = ff.embedding(input,
                              optConfig.vocab_size,
                              optConfig.word_embed_proj_dim,
                              AGGR_MODE_NONE,
                              DT_FLOAT,
                              NULL,
                              embed_init);
  Layer *embedding = ff.layers.back();

  weights_layers.emplace("embed_tokens_weight", embedding);

  Tensor positional_embedding = ff.embedding(position_input,
                                             optConfig.max_position_embeddings,
                                             optConfig.hidden_size,
                                             AGGR_MODE_NONE,
                                             DT_FLOAT,
                                             NULL,
                                             embed_init);
  Layer *pos_embedding = ff.layers.back();
  weights_layers.emplace("embed_positions_weight", pos_embedding);

  Tensor residual = ff.add(token, positional_embedding);

  int num_transformer_layers_per_gpu = (32 + num_devices - 1) / num_devices;

  for (int i = 0; i < optConfig.num_hidden_layers; i++) {
    // 125m, 1.7B, ..., 175B applies layer norm BEFORE attention,
    // 350m applies layer norm AFTER attention
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#LL324C1-L325C1
    // this version is before normalization

    Tensor hidden_states = ff.layer_norm(
        residual, axes, optConfig.layer_norm_elementwise_affine, 1e-05);
    Layer *self_attn_layer_norm = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_self_attn_layer_norm_weight",
                           self_attn_layer_norm);
    if (i % num_transformer_layers_per_gpu == 0) {
      mapping[hidden_states].push_back(
          machine_views[i / num_transformer_layers_per_gpu]);
    }

    Tensor mha = ff.inc_multihead_self_attention(
        hidden_states,
        optConfig.hidden_size,
        optConfig.num_attention_heads,
        optConfig.hidden_size / optConfig.num_attention_heads,
        optConfig.hidden_size / optConfig.num_attention_heads,
        0.0f,
        true,
        false,
        false,
        NULL,
        false,
        /*scaling query*/ true,
        /*sacling factor*/
        pow((optConfig.hidden_size / optConfig.num_attention_heads), -0.5),
        /*qk_prod_scaling*/ false);

    Layer *attention_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_attention_weight",
                           attention_layer);
    Tensor added = ff.add(mha, residual);

    Tensor final_norm = ff.layer_norm(
        added, axes, optConfig.layer_norm_elementwise_affine, 1e-05);
    Layer *final_layer_norm = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_final_layer_norm_weight",
                           final_layer_norm);

    //--------linear fc1 fc2 ----------
    Tensor fc1 = ff.dense(final_norm, optConfig.ffn_dim, AC_MODE_NONE, true);
    Layer *fc1_linear = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_fc1_weight",
                           fc1_linear);
    Tensor activation = ff.relu(fc1, false);

    Tensor fc2 =
        ff.dense(activation, optConfig.hidden_size, AC_MODE_NONE, true);
    Layer *fc2_linear = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_fc2_weight",
                           fc2_linear);
    residual = ff.add(added, fc2);
  }

  // final
  Tensor all_final_norm = ff.layer_norm(
      residual, axes, optConfig.layer_norm_elementwise_affine, 1e-05);
  Layer *all_final_norm_layer = ff.layers.back();
  weights_layers.emplace("final_layer_norm_weight", all_final_norm_layer);

  Tensor lm_head =
      ff.dense(all_final_norm, optConfig.vocab_size, AC_MODE_NONE, false);
  Layer *lm_head_layer = ff.layers.back();
  weights_layers.emplace("embed_tokens_weight_lm_head", lm_head_layer);

  Tensor output = ff.arg_top_k(lm_head, /*k=*/1, false);
  //------------------- compile the model --------------------------------
  std::cout << "------start compile ----------" << std::endl;
  InferenceManager im(ffconfig, 1, 1);
  im.compile_model_and_allocate_buffer(&ff, mapping);
  RequestManager rm;

  ParallelTensor input_pt;
  ff.get_parallel_tensor_from_tensor(input, input_pt);
  assert(im.tensor_buffer.find(input_pt) != im.tensor_buffer.end());

  ParallelTensor pos_pt;
  ff.get_parallel_tensor_from_tensor(position_input, pos_pt);
  assert(im.tensor_buffer.find(pos_pt) != im.tensor_buffer.end());

  //-------------------load weights and inputs------------------
  FileDataLoader fileloader(optConfig.input_path,
                            optConfig.weight_file_path,
                            optConfig.num_attention_heads,
                            optConfig.hidden_size,
                            optConfig.hidden_size /
                                optConfig.num_attention_heads);
  //"Today is a beautiful day and I want"
  std::vector<int> prompt = {2, 5625, 16, 10, 2721, 183, 8, 38, 236};
  rm.register_new_request(prompt, 30);
  fileloader.load_weights(&ff, weights_layers);

  im.init_operators_inference(&ff);
  int depth = 0;
  std::map<int, Future> future_handlers;
  std::map<int, BatchConfig> batch_configs;
  int sentence_length = 9;
  while (true) {
    int bid = 0;
    if (future_handlers.find(bid) == future_handlers.end()) {
      BatchConfig bc;
      InferenceResult ir;
      bc = rm.prepare_next_batch(bc, ir);
      FutureMap fm = im.inference(&ff, bid, bc);
      assert(fm.get_future_map_domain().get_volume() == 1);
      future_handlers[bid] = fm.get_future(0);
      batch_configs[bid] = bc;
    } else {
      Future future = future_handlers[bid];
      if (!future.is_ready(true /*subscribe*/)) {
        continue;
      } else {
        std::cout << "future is ready...." << std::endl;
      }
      // process end
      InferenceResult ir = future.get_result<InferenceResult>();
      BatchConfig bc = batch_configs[bid];
      bc = rm.prepare_next_batch(bc, ir);
      sentence_length += bc.num_tokens;
      FutureMap fm = im.inference(&ff, bid, bc);
      assert(fm.get_future_map_domain().get_volume() == 1);
      future_handlers[bid] = fm.get_future(0);
      batch_configs[bid] = bc;
    }
  }
  std::cout << "----------inference finished--------------" << std::endl;
}

void FlexFlow::register_custom_tasks() {}
