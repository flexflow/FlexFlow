/* Copyright 2021 Stanford University
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

#include "mlp.h"

using namespace Legion;
using namespace FlexFlow;

DataLoader::DataLoader(FFModel &ff,
                       MLPConfig const &mlpConfig,
                       InferenceManager const *im,
                       Tensor input) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  printf("Use random dataset...");

  // The number of samples is the total number of request samples that can ever
  // be loaded into memory at the same time. In the case of training, the value
  // is batchSize * workersPerNode * numNodes, since each worker can only
  // process one batch at a time. In inference,  batchSize
  int max_parallel_requests =
      im->max_num_inflight_batches *
      (ff.config.batchSize * im->max_num_requests_per_batch);
  num_samples =
      max_parallel_requests * ff.config.workersPerNode * ff.config.numNodes;
  printf("Number of random samples = %d\n", num_samples);

  // return;

  // Create full input
  {
    batch_input = input;
    int const dims[] = {num_samples,
                        mlpConfig.sequence_length * mlpConfig.embedding_size};
    full_input = ff.create_tensor<2>(dims, DT_FLOAT);
  }

  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1, TaskArgument(NULL, 0));
  launcher.add_region_requirement(
      RegionRequirement(full_input->parallel_tensor->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_input->parallel_tensor->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  runtime->execute_task(ctx, launcher);
  reset();
  // next_batch(ff);
}

void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  assert(regions.size() == 1); // no labels
  assert(task->regions.size() == 1);
  // Note that these instances are in ZCM, can only use
  // TensorAccessorW with readOutput flag
  AccessorWO<float, 2> const acc_input(regions[0], FID_DATA);
  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  float *input_ptr = acc_input.ptr(rect_input.lo);
  // Fill dataset with random data
  for (int i = 0; i < rect_input.volume(); i++) {
    input_ptr[i] = ((float)std::rand()) / RAND_MAX;
  }
  printf("finish loading data\n");
}

void DataLoader::next_batch(FFModel &ff) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Load Input
  {
    Rect<2> rect = runtime->get_index_space_domain(
        ctx, batch_input->parallel_tensor->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<2> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }

    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1,
                           batch_input->parallel_tensor->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_input->parallel_tensor->machine_view.hash());
    launcher.add_region_requirement(
        RegionRequirement(full_input->parallel_tensor->region,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          full_input->parallel_tensor->region,
                          MAP_TO_FB_MEMORY));
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
  // progress to the next_index
  next_index += ff.config.batchSize;
}

void DataLoader::reset() {
  next_index = 0;
}

Tensor create_mlp(FFModel *model,
                  MLPConfig const *mlpConfig,
                  Tensor const &input1,
                  Tensor const &input2) {
  Tensor t1 = input1, t2 = input2;
  for (int i = 0; i < mlpConfig->hidden_dims.size(); i++) {
    int const dims[] = {mlpConfig->hidden_dims[i], t1->dims[0]};
    ActiMode acti_mode =
        (i + 1 == mlpConfig->hidden_dims.size()) ? AC_MODE_NONE : AC_MODE_RELU;
    t1 = model->dense(t1, mlpConfig->hidden_dims[i], acti_mode, false);
    t2 = model->dense(t2, mlpConfig->hidden_dims[i], acti_mode, false);
  }
  Tensor t = model->add(t1, t2);
  return model->softmax(t);
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {

  // Inference parameters
  int total_requests =
      256; // total number of requests processed as part of the simulation
  int request_tensor_size = 4; // request tensor dimensions
  bool poisson_distribution = true;
  double lambda = 25; // average number of request arrivals per second
  int num_requests_per_batch = 5;
  int num_inflight_batches = 10;

  // MLP parameters
  int embedding_size = 1024;
  int sequence_length = 512;
  std::vector<int> hidden_dims = {
      8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192};

  FFConfig ffConfig;
  ffConfig.batchSize = 1;
  {
    fprintf(stderr,
            "batchSize(%d) workersPerNodes(%d) numNodes(%d)\n",
            ffConfig.batchSize,
            ffConfig.workersPerNode,
            ffConfig.numNodes);
  }
  FFModel ff(ffConfig);
  MLPConfig mlpConfig(embedding_size, sequence_length, hidden_dims);
  {
    stringstream hd;
    hd << '{';
    for (int i = 0; i < hidden_dims.size(); i++) {
      if (i != 0) {
        hd << ",";
      }
      hd << hidden_dims[i];
    }
    hd << '}';
    fprintf(stderr,
            "embedding_size(%d) sequence_length(%d) hidden_dims(%s)\n",
            mlpConfig.embedding_size,
            mlpConfig.sequence_length,
            hd.str().c_str());
  }

  Tensor input1, input2;
  {
    int const dims[] = {total_requests,
                        mlpConfig.sequence_length * mlpConfig.embedding_size};
    input1 = ff.create_tensor<2>(dims, DT_FLOAT);
    input2 = ff.create_tensor<2>(dims, DT_FLOAT);
  }
  Tensor t = create_mlp(&ff, &mlpConfig, input1, input2);

  InferenceManager im(&ff, num_requests_per_batch, num_inflight_batches);
  ff.init_operators();

  // Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();

  ///////////////////////////////////////////////////////////////////////////////////

  // Main loop, processing requests as they come (from the generator)
  int index = 0;
  int processed_requests = 0;
  Generator data_generator(
      total_requests, request_tensor_size, poisson_distribution, lambda);
  while (processed_requests < total_requests) {
    vector<vector<double>> req = data_generator.get_requests();
    int iterations = req.size();
    for (int iter = 0; iter < iterations; iter++) {
      runtime->begin_trace(ctx, 111 /*trace_id*/);
      im.inference((index++) % num_inflight_batches);
      runtime->end_trace(ctx, 111 /*trace_id*/);
    }
    processed_requests += iterations;
  }

  ///////////////////////////////////////////////////////////////////////////////////

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
         ffConfig.batchSize * 128 * ffConfig.epochs / run_time);
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
    TaskVariantRegistrar registrar(FlexFlow::CUSTOM_GPU_TASK_ID_1,
                                   "Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Input Task");
  }
}
