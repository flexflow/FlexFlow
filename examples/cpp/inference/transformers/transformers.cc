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

#include "transformers.h"
#include "flexflow/inference.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace Legion;

LegionRuntime::Logger::Category log_app("Transformers");

void parse_input_args(char **argv, int argc, TransformerConfig &config) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--dataset")) {
      config.dataset_path = std::string(argv[++i]);
      continue;
    }
  }
}

Tensor create_inc_multihead_attention_decoder(FFModel *model,
                                              TransformerConfig const *transformerConfig,
                                              Tensor const &input) {
  std::vector<int> axes{1};
  Tensor t = model->inc_multihead_self_attention(input,
                                                 transformerConfig->hidden_size,
                                                 transformerConfig->num_attention_heads,
                                                 transformerConfig->attention_kdim,
                                                 transformerConfig->attention_vdim);

  t = model->layer_norm(model->add(t, input), axes, true, 1e-05);
  Tensor x = model->dense(
      model->dense(t, transformerConfig->hidden_size, AC_MODE_RELU, false /*bias*/),
      transformerConfig->hidden_size,
      AC_MODE_NONE,
      false /*bias*/);
  t = model->layer_norm(model->add(x, t), axes, true, 1e-05);
  return t;
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  //----------------------- Initial configurations ------------------------
  TransformerConfig transformerConfig;
  FFConfig ffConfig;
  ffConfig.batchSize = transformerConfig.batch_size;
  {
    InputArgs const &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, transformerConfig);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
                  ffConfig.batchSize,
                  ffConfig.workersPerNode,
                  ffConfig.numNodes);
  }
  FFModel ff(ffConfig);

  //----------------------- Create inputs --------------------------------
  Tensor input;
  {
    int const dims[] = {BatchConfig::MAX_NUM_TOKENS, transformerConfig.token_dim};
    input = ff.create_tensor<2>(dims, DT_FLOAT);
  }

  //----------------------- Define the model ------------------------------
  Tensor t = input;
  for (int i = 0; i < transformerConfig.num_layers; i++) {
    t = create_inc_multihead_attention_decoder(&ff, &transformerConfig, input);
  }
  t = ff.dense(t, transformerConfig.out_dim, AC_MODE_RELU);
  t = ff.softmax(t);

  //------------------- Initialize the inference manager ------------------
  InferenceManager im(
      &ff, transformerConfig.batch_size, transformerConfig.num_inflight_batches);
  im.compile_model_and_allocate_buffer();
  im.init_operators_inference();

  //------------ Initialize the data loader and data generator ------------
  DataGenerator data_generator(transformerConfig.total_requests,
                               transformerConfig.token_dim,
                               transformerConfig.sequence_length,
                               transformerConfig.poisson_distribution,
                               transformerConfig.arrival_rate);
  ParallelTensor input_pt;
  ff.get_parallel_tensor_from_tensor(input, input_pt);
  DataLoader data_loader(ff, transformerConfig, data_generator, input_pt);

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
  std::cout << im.max_tokens_per_batch << std::endl;
  std::pair<size_t, size_t> new_prompts;
  BatchConfig *bc = nullptr;
  
  // simulation loop. For deployment, we will use a while(true)
  while (processed_requests < transformerConfig.total_requests) {
    for (int bid = 0; bid < im.max_inflight_batches; bid++) {
      if (future_handlers.find(bid) == future_handlers.end()) {
        new_prompts =
            data_generator.get_requests(im.max_tokens_per_batch);
        assert(new_prompts.second < BatchConfig::MAX_NUM_REQUESTS);
        bc = new BatchConfig();
      } else {
        Future future = future_handlers[bid];
        if (!future.is_ready(true /*subscribe*/)) {
          continue;
        }
        InferenceResult ir = future.get_result<InferenceResult>();
        bc = batch_configs[bid];
        processed_requests += bc->update_results(ir);
        size_t available_slots = im.max_tokens_per_batch - bc->num_active_tokens();
        new_prompts = data_generator.get_requests(available_slots);
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
         transformerConfig.total_requests / run_time);
}
