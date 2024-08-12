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

#include "transformer.h"

using namespace Legion;

Realm::Logger log_app("Transformer");

Tensor create_emb(FFModel *model,
                  Tensor const &input,
                  int input_dim,
                  int output_dim,
                  int idx) {
  float range = sqrt(1.0f / input_dim);
  Initializer *embed_init = new UniformInitializer(std::rand(), -range, range);
  return model->embedding(
      input, input_dim, output_dim, AGGR_MODE_SUM, DT_FLOAT, NULL, embed_init);
}

Tensor create_attention_encoder(FFModel *model,
                                Tensor const &input,
                                int hidden_dim,
                                int num_heads,
                                int kdim,
                                int vdim) {
  Tensor t = model->multihead_attention(
      input, input, input, hidden_dim, num_heads, kdim, vdim);
  return model->dense(model->dense(t, hidden_dim, AC_MODE_RELU, false /*bias*/),
                      hidden_dim,
                      AC_MODE_NONE,
                      false /*bias*/);
}

void create_attention_encoder_decoder(FFModel *model,
                                      Tensor const &input1,
                                      Tensor const &input2,
                                      Tensor &output1,
                                      Tensor &output2,
                                      int hidden_dim,
                                      int num_heads,
                                      int kdim,
                                      int vdim) {
  Tensor t1 =
      model->add(model->multihead_attention(
                     input1, input1, input1, hidden_dim, num_heads, kdim, vdim),
                 input1);
  t1 = model->dense(model->dense(t1, hidden_dim, AC_MODE_RELU, false /*bias*/),
                    hidden_dim,
                    AC_MODE_NONE,
                    false /*bias*/);
  Tensor t2 =
      model->add(model->multihead_attention(
                     input2, input2, input2, hidden_dim, num_heads, kdim, vdim),
                 input2);
  t2 = model->add(
      model->multihead_attention(t2, t1, t1, hidden_dim, num_heads, kdim, vdim),
      t2);
  t2 = model->dense(model->dense(t2, hidden_dim, AC_MODE_RELU, false /*bias*/),
                    hidden_dim,
                    AC_MODE_NONE,
                    false /*bias*/);
  output1 = t1;
  output2 = t2;
}

TransformerConfig::TransformerConfig(void) {
  hidden_size = 1024;
  embedding_size = 1024;
  num_heads = 16;
  num_layers = 12;
  sequence_length = 512;
}

void parse_input_args(char **argv, int argc, TransformerConfig &config) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--num-layers")) {
      config.num_layers = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--embedding-size")) {
      config.embedding_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--hidden-size")) {
      config.hidden_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--num-heads")) {
      config.num_heads = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--sequence-length")) {
      config.sequence_length = atoi(argv[++i]);
      continue;
    }
  }
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffConfig;
  TransformerConfig tfConfig;
  {
    InputArgs const &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, tfConfig);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
                  ffConfig.batchSize,
                  ffConfig.workersPerNode,
                  ffConfig.numNodes);
    log_app.print("Hidden Size(%d)", tfConfig.hidden_size);
    log_app.print("Embedding Vocab Size(%d)", tfConfig.embedding_size);
    log_app.print("Number of Heads(%d)", tfConfig.num_heads);
    log_app.print("Number of Layers(%d)", tfConfig.num_layers);
    log_app.print("Sequence Length(%d)", tfConfig.sequence_length);
  }
  FFModel ff(ffConfig);
  Tensor input;
  {
    int const dims[] = {
        ffConfig.batchSize, tfConfig.sequence_length, tfConfig.hidden_size};
    input = ff.create_tensor<3>(dims, DT_FLOAT);
  }
  // Tensor t = create_emb(&ff, input, tfConfig.embedding_size,
  // tfConfig.hidden_size); Tensor input1 = input, input2 = input; Tensor t1,
  // t2;
  Tensor t = input;
  for (int i = 0; i < tfConfig.num_layers; i++) {
    t = create_attention_encoder(&ff,
                                 t,
                                 tfConfig.hidden_size,
                                 tfConfig.num_heads,
                                 tfConfig.hidden_size / tfConfig.num_heads,
                                 tfConfig.hidden_size / tfConfig.num_heads);
    // create_attention_encoder_decoder(&ff, input1, input2, t1, t2,
    //     tfConfig.hidden_size, tfConfig.num_heads,
    //     tfConfig.hidden_size / tfConfig.num_heads,
    //     tfConfig.hidden_size / tfConfig.num_heads);
    // input1 = t1;
    // input2 = t2;
  }
  t = ff.dense(t, 1, AC_MODE_NONE, false /*bias*/);
  Optimizer *optimizer = new SGDOptimizer(&ff, 0.01f);
  std::vector<MetricsType> metrics;
  // metrics.push_back(METRICS_ACCURACY);
  // metrics.push_back(METRICS_MEAN_SQUARED_ERROR);
  ff.compile(optimizer, LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE, metrics);
  // Data Loader
  DataLoader loader(ff, tfConfig, input, ff.label_tensor);
  loader.next_batch(ff);
  loader.reset();
  ff.init_operators();

  // Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  log_app.print("Warmup finished...Start timer...");
  log_app.print("Num. epochs = %d", ffConfig.epochs);
  log_app.print("Num. iterations/epoch = %d",
                loader.num_samples / ffConfig.batchSize);
  printf("parameters.size() = %lu\n", ff.parameters.size());
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    loader.reset();
    ff.reset_metrics();
    int iterations = loader.num_samples / ffConfig.batchSize;
    for (int iter = 0; iter < iterations; iter++) {
      // Only load data once for random input
      if (iter == 0 && epoch == 0) {
        loader.next_batch(ff);
      }
      // runtime->begin_trace(ctx, 111 /*trace_id*/);
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
      // runtime->end_trace(ctx, 111 /*trace_id*/);
    }
  }
  // End timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double run_time = 1e-6 * (ts_end - ts_start);
  printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n",
         run_time,
         loader.num_samples * ffConfig.epochs / run_time);
}

DataLoader::DataLoader(FFModel &ff,
                       TransformerConfig const &tf,
                       Tensor const &_input,
                       Tensor const &_label) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  num_samples = 0;
  log_app.print("Use random dataset...");
  num_samples =
      ff.config.batchSize * ff.config.workersPerNode * ff.config.numNodes;
  log_app.print("Number of random samples = %d\n", num_samples);
  return;
  {
    batch_input = _input;
    int const dims[] = {num_samples, tf.sequence_length, tf.hidden_size};
    full_input = ff.create_tensor<3>(dims, DT_FLOAT);
  }
  {
    batch_label = _label;
    int const dims[] = {num_samples, tf.sequence_length, 1};
    full_label = ff.create_tensor<3>(dims, DT_FLOAT);
  }
  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1, TaskArgument(NULL, 0));
  // regions[0]: full_sparse_input
  launcher.add_region_requirement(
      RegionRequirement(full_input->parallel_tensor->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_input->parallel_tensor->region,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label->parallel_tensor->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_label->parallel_tensor->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  runtime->execute_task(ctx, launcher);
}

void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  // Note that these instances are in ZCM, can only use
  // TensorAccessorW with readOutput flag
  AccessorWO<float, 3> const acc_input(regions[0], FID_DATA);
  AccessorWO<float, 3> const acc_label(regions[1], FID_DATA);
  Rect<3> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<3> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  float *input_ptr = acc_input.ptr(rect_input.lo);
  float *label_ptr = acc_label.ptr(rect_label.lo);
  // assert(rect_input == rect_label);

  for (size_t i = 0; i < rect_input.volume(); i++) {
    input_ptr[i] = ((float)std::rand()) / RAND_MAX;
  }
  for (size_t i = 0; i < rect_label.volume(); i++) {
    label_ptr[i] = std::rand() % 2;
  }
}

void DataLoader::next_batch(FFModel &ff) {
  return;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Load Input
  {
    Domain domain = runtime->get_index_space_domain(
        ctx, batch_input->parallel_tensor->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % batch_input->parallel_tensor->dims[2].size ==
             0);
      meta.num_samples =
          ff.config.batchSize / batch_input->parallel_tensor->dims[2].size;
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2,
                           batch_input->parallel_tensor->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_input->parallel_tensor->machine_view.hash());
    // Full dataset in ZCM
    launcher.add_region_requirement(
        RegionRequirement(full_input->parallel_tensor->region,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          full_input->parallel_tensor->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_input->parallel_tensor->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_input->parallel_tensor->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load Labels
  {
    Domain domain = runtime->get_index_space_domain(
        ctx, batch_label->parallel_tensor->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % batch_label->parallel_tensor->dims[2].size ==
             0);
      meta.num_samples =
          ff.config.batchSize / batch_label->parallel_tensor->dims[2].size;
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2,
                           batch_label->parallel_tensor->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_label->parallel_tensor->machine_view.hash());
    // Full dataset in ZCM
    launcher.add_region_requirement(
        RegionRequirement(full_label->parallel_tensor->region,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          full_label->parallel_tensor->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_label->parallel_tensor->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_label->parallel_tensor->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // progress next_index
  next_index += ff.config.batchSize;
}

void DataLoader::reset() {
  next_index = 0;
}

void FlexFlow::register_custom_tasks() {
  // Load entire dataset
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_1, "Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_entire_dataset>(
        registrar, "Load Entire Dataset Task");
  }
  // Load input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2, "Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Inputs Task");
  }
}
