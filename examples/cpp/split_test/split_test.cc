#include "flexflow/model.h"

using namespace Legion;
using namespace FlexFlow;

Realm::Logger log_app("split_test");

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  int layer_dims[4] = {256, 128, 64, 32};

  FFConfig ffConfig;
  log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
                ffConfig.batchSize,
                ffConfig.workersPerNode,
                ffConfig.numNodes);
  FFModel ff(ffConfig);

  Tensor input;
  {
    int const dims[] = {1, ffConfig.batchSize, layer_dims[0]};
    input = ff.create_tensor<3>(dims, DT_FLOAT);
    log_app.print("input size: %d %d %d", dims[0], dims[1], dims[2]);
  }

  Tensor t, t1, t2;

  t = input;
  t = ff.dense(input, layer_dims[1]);
  t = ff.relu(t);
  t1 = ff.dense(t, layer_dims[2]);
  t2 = ff.dense(t, layer_dims[2]);
  t = ff.add(t1, t2);
  t = ff.relu(t);
  t1 = ff.dense(t, layer_dims[3]);
  t2 = ff.dense(t, layer_dims[3]);
  t = ff.add(t1, t2);
  t = ff.relu(t);
  t = ff.softmax(t);

  Optimizer *optimizer = new SGDOptimizer(&ff, 0.001f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);
  ff.init_operators();
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    ff.reset_metrics();
    int iterations = 128; // data_loader.num_samples / ffConfig.batchSize;

    for (int iter = 0; iter < iterations; iter++) {
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
         128 * ffConfig.batchSize * ffConfig.epochs / run_time);
}

void FlexFlow::register_custom_tasks() {}
