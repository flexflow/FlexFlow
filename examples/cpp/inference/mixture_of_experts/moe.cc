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
                       moeConfig->alpha);
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

Tensor create_moe_encoder(FFModel *model,
                          MoeConfig const *moeConfig,
                          Tensor const &input) {
  std::vector<int> axes = {0, 1, 2};
  Tensor x = input;
  for (int i = 0; i < moeConfig->num_encoder_layers; i++) {
    x = model->layer_norm(
        model->add(model->multihead_attention(x,
                                              x,
                                              x,
                                              moeConfig->hidden_size,
                                              moeConfig->num_attention_heads,
                                              moeConfig->attention_kdim,
                                              moeConfig->attention_vdim),
                   x),
        axes,
        true,
        1e-05);
    x = model->layer_norm(
        model->add(create_moe(model, moeConfig, x), x), axes, true, 1e-05);
  }
  return x;
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  //----------------------- Initial configurations ------------------------
  MoeConfig moeConfig;
  FFConfig ffConfig;
  ffConfig.batchSize = moeConfig.batch_size;
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
    int const dims[] = {
        ffConfig.batchSize, moeConfig.sequence_length, moeConfig.token_dim};
    input = ff.create_tensor<3>(dims, DT_FLOAT);
  }

  //----------------------- Define the model ------------------------------
  Tensor t = create_moe_encoder(&ff, &moeConfig, input);
  // Tensor t = create_moe(&ff, &moeConfig, input);
  t = ff.dense(t, moeConfig.out_dim, AC_MODE_RELU);

  //------------------- Initialize the inference manager ------------------
  InferenceManager im(
      &ff, moeConfig.batch_size, moeConfig.num_inflight_batches);
  im.compile_model_and_allocate_buffer();
  im.init_operators_inference();

  //------------ Initialize the data loader and data generator ------------
  DataGenerator data_generator(moeConfig.total_requests,
                               moeConfig.token_dim,
                               moeConfig.sequence_length,
                               moeConfig.poisson_distribution,
                               moeConfig.arrival_rate);
  ParallelTensor input_pt;
  ff.get_parallel_tensor_from_tensor(input, input_pt);
  DataLoader data_loader(ff, moeConfig, data_generator, input_pt);

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
  int num_devices = ffConfig.workersPerNode * ffConfig.numNodes;
  data_generator.start_timer();
  std::map<int, Future> future_handlers;
  std::map<int, BatchConfig *> batch_configs;
  assert(im.max_num_requests_per_batch <= BatchConfig::MAX_NUM_REQUESTS);
  std::pair<size_t, size_t> new_prompts;
  BatchConfig *bc = nullptr;
  while (processed_requests < moeConfig.total_requests) {
    for (int bid = 0; bid < im.max_num_inflight_batches; bid++) {
      if (future_handlers.find(bid) == future_handlers.end()) {
        new_prompts =
            data_generator.get_requests(im.max_num_requests_per_batch);
        assert(new_prompts.second < im.max_num_requests_per_batch);
        bc = new BatchConfig();
      } else {
        Future future = future_handlers[bid];
        if (!future.is_ready(true /*subscribe*/)) {
          continue;
        }
        InferenceResult ir = future.get_result<InferenceResult>();
        bc = batch_configs[bid];
        processed_requests += bc->update_results(ir);
        size_t available_slots =
            im.max_num_requests_per_batch - bc->num_active_requests();
        new_prompts =
            data_generator.get_requests(im.max_num_requests_per_batch);
      }
      for (size_t i = 0; i < new_prompts.second; i++) {
        size_t guid = new_prompts.first + i;
        ssize_t seq_len = data_generator.get_request_length(guid);
        assert(seq_len >= 0);
        assert(bc->register_new_request(guid, (size_t)seq_len));
      }
      bc->prepare_next_batch();
      // TODO: loading data
      data_loader.next_batch(ff, bc);

      runtime->begin_trace(ctx, 111 + bid % num_devices /*trace_id*/);
      FutureMap fm = im.inference(bid, *bc);
      runtime->end_trace(ctx, 111 + bid % num_devices /*trace_id*/);
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
