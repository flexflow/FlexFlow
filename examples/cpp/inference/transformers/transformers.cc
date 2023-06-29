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

Tensor create_inc_multihead_attention_decoder(
    FFModel *model,
    TransformerConfig const *transformerConfig,
    Tensor const &input) {
  std::vector<int> axes{0};
  Tensor t =
      transformerConfig->incremental_mode
          ? model->inc_multihead_self_attention(
                input,
                transformerConfig->hidden_size,
                transformerConfig->num_attention_heads,
                transformerConfig->attention_kdim,
                transformerConfig->attention_vdim)
          : model->multihead_attention(input,
                                       input,
                                       input,
                                       transformerConfig->hidden_size,
                                       transformerConfig->num_attention_heads,
                                       transformerConfig->attention_kdim,
                                       transformerConfig->attention_vdim);
  t = model->layer_norm(model->add(t, input), axes, true, 1e-05);
  Tensor x = model->dense(
      model->dense(
          t, transformerConfig->hidden_size, AC_MODE_RELU, false /*bias*/),
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
    int const dims[] = {ffConfig.batchSize, transformerConfig.sequence_length};
    input = ff.create_tensor<2>(dims, DT_INT32);
  }

  //----------------------- Define the model ------------------------------
  Tensor t = input;

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);
  t = ff.embedding(t,
                   transformerConfig.vocab_size,
                   transformerConfig.token_dim,
                   AGGR_MODE_NONE,
                   DT_FLOAT,
                   NULL,
                   embed_init);

  for (int i = 0; i < transformerConfig.num_layers; i++) {
    t = create_inc_multihead_attention_decoder(&ff, &transformerConfig, t);
  }
  t = ff.dense(t, transformerConfig.out_dim, AC_MODE_RELU);
  t = ff.softmax(t);
  // select most likely next token
  Tensor output = ff.arg_top_k(t, /*k=*/1, false);

  //------------------- Initialize the inference manager ------------------
  InferenceManager im(ff.config, transformerConfig.batch_size);
  std::unordered_map<Tensor, std::vector<MachineView>> mapping;
  im.compile_model_and_allocate_buffer(&ff, mapping);
  im.init_operators_inference(&ff);

  //------------ Initialize the data loader and data generator ------------
  /* size_t min_input_tokens = 32, max_input_tokens = 512,
         min_tokens_to_generate = 1, max_tokens_to_generate = 128; */
  size_t min_input_tokens = 5, max_input_tokens = 10,
         min_tokens_to_generate = 1,
         max_tokens_to_generate = MAX_SEQ_LEN - max_input_tokens;
  DataGenerator data_generator(transformerConfig.total_requests,
                               transformerConfig.vocab_size,
                               min_input_tokens,
                               max_input_tokens,
                               min_tokens_to_generate,
                               max_tokens_to_generate,
                               transformerConfig.poisson_distribution,
                               transformerConfig.arrival_rate);
  ParallelTensor input_pt;
  ff.get_parallel_tensor_from_tensor(input, input_pt);
  assert(im.tensor_buffer.find(input_pt) != im.tensor_buffer.end());
  assert(im.tensor_buffer[input_pt].size() == ffConfig.data_parallelism_degree);
  DataLoader data_loader(
      ff, transformerConfig, data_generator, im.tensor_buffer[input_pt]);

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
  std::pair<size_t, size_t> new_prompts;
  BatchConfig *bc = nullptr;
  std::map<size_t, int> batch_predictions[ffConfig.data_parallelism_degree];

  assert(im.max_num_tokens_per_batch == transformerConfig.batch_size);
  // assert(transformerConfig.batch_size <= BatchConfig::MAX_NUM_REQUESTS);

  // simulation loop. For deployment, we will use a while(true)
  while (processed_requests < transformerConfig.total_requests) {
    for (int bid = 0; bid < ffConfig.data_parallelism_degree; bid++) {
      size_t max_reqs, max_tkns;
      if (future_handlers.find(bid) == future_handlers.end()) {
        max_reqs = transformerConfig.incremental_mode
                       ? bc->MAX_NUM_REQUESTS
                       : im.max_num_tokens_per_batch;
        max_tkns =
            transformerConfig.sequence_length * transformerConfig.batch_size;
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
        processed_requests += bc->update_results(&ir);
        max_reqs = transformerConfig.incremental_mode
                       ? bc->MAX_NUM_REQUESTS - bc->num_active_requests()
                       : im.max_num_tokens_per_batch;
        max_tkns =
            transformerConfig.sequence_length * transformerConfig.batch_size -
            (transformerConfig.incremental_mode ? bc->num_active_tokens() : 0);
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
      FutureMap fm = im.inference(&ff, bid, *bc);
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
         transformerConfig.total_requests / run_time);
}
