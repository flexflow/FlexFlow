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

#include "mlp.h"
#include "data_generator.h"
#include "flexflow/inference.h"
#include <fstream>
#include <sstream>
#include <string>

using namespace Legion;
using namespace FlexFlow;


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
  ffConfig.batchSize=1;
  {
    fprintf(stderr, "batchSize(%d) workersPerNodes(%d) numNodes(%d)\n",
      ffConfig.batchSize,
      ffConfig.workersPerNode,
      ffConfig.numNodes
    );
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
    fprintf(stderr, "embedding_size(%d) sequence_length(%d) hidden_dims(%s)\n", mlpConfig.embedding_size, mlpConfig.sequence_length, hd.str().c_str());
  }
  
  Tensor input1, input2;
  {
    int const dims[] = {total_requests, mlpConfig.sequence_length * mlpConfig.embedding_size};
    input1 = ff.create_tensor<2>(dims, DT_FLOAT);
    input2 = ff.create_tensor<2>(dims, DT_FLOAT);
  }
  Tensor t = create_mlp(&ff, &mlpConfig, input1, input2);
  
  InferenceManager im(&ff, num_requests_per_batch, num_inflight_batches);
  // im.compile_model_and_allocate_buffer();
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
    processed_requests+= iterations;
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

void FlexFlow::register_custom_tasks() {}
