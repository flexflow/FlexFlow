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

#include "moe.h"
#include "flexflow/inference.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace Legion;

LegionRuntime::Logger::Category log_app("MoE");

void parse_input_args(char **argv, int argc, MoeConfig &config) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--dataset")) {
      config.dataset_path = std::string(argv[++i]);
      continue;
    }
  }
}

void MoeConfig::load_configs() {
  std::string folder =
      "/home/ubuntu/nlp_gpt3_text-generation_0.35B_MoE-64/model/c-models/1-gpu";
  std::string config_ini_filepath = folder + "/config.ini";
  std::map<std::string, std::string> conf = get_configs(config_ini_filepath);
  num_exp = std::min(MAX_EXPERTS, std::stoi(conf["expert_num"]));
  int tensor_para_size = std::stoi(conf["tensor_para_size"]);
  assert(num_exp % tensor_para_size == 0);
  experts_per_block = num_exp / tensor_para_size;
  num_layers = std::min(MAX_LAYERS, std::stoi(conf["num_layer"]));
  vocab_size = std::stoi(conf["vocab_size"]);
  num_attention_heads = std::stoi(conf["head_num"]);
  int gpt_with_moe = std::stoi(conf["gpt_with_moe"]);
  assert(gpt_with_moe == 1);
  moe_layers.clear();
  moe_layers = setFromList(conf["moe_layers"]);
  max_sequence_length =
      std::min(MAX_SEQ_LEN, std::stoi(conf["max_pos_seq_len"]));
  assert(max_sequence_length <= MAX_SEQ_LEN);
  size_per_head = std::stoi(conf["size_per_head"]);
  hidden_size = token_dim = out_dim = size_per_head * num_attention_heads;
  assert(hidden_size <= DATA_DIM);
}

void MoeConfig::print_configs() {
  std::cout << "token_dim: " << token_dim << std::endl;
  std::cout << "max_sequence_length: " << max_sequence_length << std::endl;
  std::cout << "batch_size: " << batch_size << std::endl;
  std::cout << "out_dim: " << out_dim << std::endl;
  std::cout << "num_layers: " << num_layers << std::endl;
  std::cout << "vocab_size: " << vocab_size << std::endl;
  std::cout << "dataset_path: " << dataset_path << std::endl;
  std::cout << "total_requests: " << total_requests << std::endl;
  std::cout << "poisson_distribution: "
            << (poisson_distribution ? "true" : "false") << std::endl;
  std::cout << "arrival_rate: " << arrival_rate << std::endl;
  std::cout << "num_inflight_batches: " << num_inflight_batches << std::endl;
  std::cout << "incremental_mode: " << (incremental_mode ? "true" : "false")
            << std::endl;
  std::cout << "hidden_size: " << hidden_size << std::endl;
  std::cout << "num_attention_heads: " << num_attention_heads << std::endl;
  std::cout << "size_per_head: " << size_per_head << std::endl;
  std::cout << "num_exp: " << num_exp << std::endl;
  std::cout << "experts_per_block: " << experts_per_block << std::endl;
  std::cout << "num_select: " << num_select << std::endl;
  std::cout << "alpha: " << alpha << std::endl;
  std::cout << "lambda: " << lambda << std::endl;
  std::cout << "moe_layers: ";
  for (auto const &layer : moe_layers) {
    std::cout << layer << " ";
  }
  std::cout << std::endl;
}

Tensor create_moe(FFModel *model,
                  MoeConfig const *moeConfig,
                  Tensor const &input) {
  // MoE model
  Tensor gate_preds = model->dense(input, moeConfig->num_exp, AC_MODE_RELU);
  Tensor topK_output[2];
  model->top_k(gate_preds, topK_output, moeConfig->num_select, false);

  assert(moeConfig->num_exp % moeConfig->experts_per_block == 0);
  int nblocks = moeConfig->num_exp / moeConfig->experts_per_block;
  Tensor exp_preds;
  Tensor expert_block_inputs[3] = {input, topK_output[1], topK_output[0]};
  for (int i = 0; i < nblocks /*number of experts layers*/; i++) {
    Tensor block_preds =
        model->experts(expert_block_inputs,
                       moeConfig->experts_per_block,     /*number of experts*/
                       moeConfig->experts_per_block * i, /*expert start index*/
                       moeConfig->hidden_size,           /*output_size*/
                       moeConfig->alpha,
                       2,
                       moeConfig->hidden_size * 4);
    assert(block_preds != nullptr);
    if (i == 0) {
      exp_preds = block_preds;
    } else {
      assert(exp_preds != nullptr);
      model->add(exp_preds, block_preds, /*inplace_a*/ true);
    }
  }

  // model->get_metrics();
  return exp_preds;
}

Tensor
    gpt_moe(FFModel *model, MoeConfig const *moeConfig, Tensor const &input) {
  std::vector<int> axes = {0};
  Tensor x = input;
  for (int i = 0; i < moeConfig->num_layers; i++) {
    Tensor t = moeConfig->incremental_mode
                   ? model->inc_multihead_self_attention(
                         x,
                         moeConfig->hidden_size,
                         moeConfig->num_attention_heads,
                         moeConfig->size_per_head,
                         moeConfig->size_per_head)
                   : model->multihead_attention(x,
                                                x,
                                                x,
                                                moeConfig->hidden_size,
                                                moeConfig->num_attention_heads,
                                                moeConfig->size_per_head,
                                                moeConfig->size_per_head);
    x = model->layer_norm(model->add(t, x), axes, true, 1e-05);
    x = model->layer_norm(
        model->add(
            (moeConfig->moe_layers.find(i) != moeConfig->moe_layers.end())
                ? create_moe(model, moeConfig, x)
                : model->dense(model->dense(x, 4 * moeConfig->hidden_size),
                               moeConfig->hidden_size),
            x),
        axes,
        true,
        1e-05);
  }
  return x;
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  //----------------------- Initial configurations ------------------------
  MoeConfig moeConfig;
  moeConfig.load_configs();

  FFConfig ffConfig;
  ffConfig.batchSize = moeConfig.batch_size;
  int num_devices = ffConfig.workersPerNode * ffConfig.numNodes;
  // overwrite experts_per_block depending on the number of devices
  assert(moeConfig.num_exp % num_devices == 0);
  moeConfig.experts_per_block = moeConfig.num_exp / num_devices;
  moeConfig.print_configs();
  {
    InputArgs const &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, moeConfig);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
                  ffConfig.batchSize,
                  ffConfig.workersPerNode,
                  ffConfig.numNodes);
  }
  FFModel ff(ffConfig);

  //----------------------- Create inputs --------------------------------
  Tensor input;
  {
    int const dims[] = {ffConfig.batchSize, moeConfig.max_sequence_length};
    input = ff.create_tensor<2>(dims, DT_INT32);
  }
  Tensor t = input;
  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);
  t = ff.embedding(t,
                   moeConfig.vocab_size,
                   moeConfig.token_dim,
                   AGGR_MODE_NONE,
                   DT_FLOAT,
                   NULL,
                   embed_init);

  //----------------------- Define the model ------------------------------
  t = gpt_moe(&ff, &moeConfig, t);
  // Tensor t = create_moe(&ff, &moeConfig, input);
  t = ff.dense(t, moeConfig.out_dim, AC_MODE_RELU);
  t = ff.softmax(t);
  // select most likely next token
  Tensor output = ff.arg_top_k(t, /*k=*/1, /*sorted=*/false);

  //------------------- Initialize the inference manager ------------------
  InferenceManager im(
      &ff, moeConfig.batch_size, moeConfig.num_inflight_batches);
  im.compile_model_and_allocate_buffer();
  im.init_operators_inference();

  //------------ Initialize the data loader and data generator ------------
  /* size_t min_input_tokens = 32, max_input_tokens = 512,
         min_tokens_to_generate = 1, max_tokens_to_generate = 128; */
  size_t min_input_tokens = 8, max_input_tokens = 128,
         min_tokens_to_generate = 1,
         max_tokens_to_generate = MAX_SEQ_LEN - max_input_tokens;
  DataGenerator data_generator(moeConfig.total_requests,
                               moeConfig.vocab_size,
                               min_input_tokens,
                               max_input_tokens,
                               min_tokens_to_generate,
                               max_tokens_to_generate,
                               moeConfig.poisson_distribution,
                               moeConfig.arrival_rate);
  ParallelTensor input_pt;
  ff.get_parallel_tensor_from_tensor(input, input_pt);
  assert(im.tensor_buffer.find(input_pt) != im.tensor_buffer.end());
  assert(im.tensor_buffer[input_pt].size() == im.max_num_inflight_batches);
  DataLoader data_loader(
      ff, moeConfig, data_generator, im.tensor_buffer[input_pt]);

  //----------------------- Start timer -----------------------------------
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();

  //----------------------- Begin inference! -------------------------------
  int index = 0;
  int processed_requests = 0;
  data_generator.start_timer();
  std::map<int, Future> future_handlers;
  std::map<int, BatchConfig *> batch_configs;
  std::pair<size_t, size_t> new_prompts;
  BatchConfig *bc = nullptr;
  std::map<size_t, int> batch_predictions[im.max_num_inflight_batches];

  assert(im.max_num_requests_per_batch == moeConfig.batch_size);

  // simulation loop. For deployment, we will use a while(true)
  while (processed_requests < moeConfig.total_requests) {
    for (int bid = 0; bid < im.max_num_inflight_batches; bid++) {
      size_t max_reqs, max_tkns;
      if (future_handlers.find(bid) == future_handlers.end()) {
        max_reqs = moeConfig.incremental_mode ? bc->MAX_NUM_REQUESTS
                                              : im.max_num_requests_per_batch;
        max_tkns = moeConfig.max_sequence_length * moeConfig.batch_size;
        new_prompts = data_generator.get_requests(max_reqs, max_tkns);
        bc = new BatchConfig();
      } else {
        Future future = future_handlers[bid];
        if (!future.is_ready(true /*subscribe*/)) {
          continue;
        }
        InferenceResult ir = future.get_result<InferenceResult>();
        bc = batch_configs[bid];
        data_loader.store_outputs(bc, ir, batch_predictions[bid]);
        processed_requests += bc->update_results(ir);
        max_reqs = moeConfig.incremental_mode
                       ? bc->MAX_NUM_REQUESTS - bc->num_active_requests()
                       : im.max_num_requests_per_batch;
        max_tkns = moeConfig.max_sequence_length * moeConfig.batch_size -
                   (moeConfig.incremental_mode ? bc->num_active_tokens() : 0);
        new_prompts = data_generator.get_requests(max_reqs, max_tkns);
      }
      assert(new_prompts.second <= max_reqs);
      if (bc->num_active_tokens() == 0 && new_prompts.second == 0) {
        continue;
      }
      for (size_t i = 0; i < new_prompts.second; i++) {
        size_t guid = new_prompts.first + i;
        std::pair<size_t, size_t> seq_lens =
            data_generator.get_request_length(guid);
        assert(seq_lens.first >= min_input_tokens &&
               seq_lens.first <= max_input_tokens &&
               seq_lens.second >= min_tokens_to_generate &&
               seq_lens.second <= max_tokens_to_generate);
        assert(bc->register_new_request(guid, seq_lens.first, seq_lens.second));
      }
      bc->prepare_next_batch();
      MachineView *view = im.get_machine_view(bid % im.num_devices);

      // runtime->begin_trace(ctx, 111 + bid % num_devices /*trace_id*/);
      data_loader.next_batch(ff, bid, bc, batch_predictions[bid], view);
      FutureMap fm = im.inference(bid, *bc);
      // runtime->end_trace(ctx, 111 + bid % num_devices /*trace_id*/);

      assert(fm.get_future_map_domain().get_volume() == 1);
      future_handlers[bid] = fm.get_future(0);
      batch_configs[bid] = bc;
    }
  }
  //----------------------- End of inference! ------------------------------

  //----------------------- Stop timer -------------------------------------
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double run_time = 1e-6 * (ts_end - ts_start);
  printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f requests/s\n",
         run_time,
         moeConfig.total_requests / run_time);
}
