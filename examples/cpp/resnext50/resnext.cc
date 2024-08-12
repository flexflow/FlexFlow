#include "flexflow/model.h"

using namespace Legion;
using FlexFlow::FFConfig;
using FlexFlow::FFModel;
using FlexFlow::Optimizer;
using FlexFlow::SGDOptimizer;
using FlexFlow::Tensor;

Realm::Logger log_app("resnext");

Tensor resnext_block(FFModel &ff,
                     Tensor input,
                     int stride_h,
                     int stride_w,
                     int out_channels,
                     int groups,
                     bool has_residual = false) {
  Tensor t = ff.conv2d(input, out_channels, 1, 1, 1, 1, 0, 0, AC_MODE_RELU);

  t = ff.conv2d(
      t, out_channels, 3, 3, stride_h, stride_w, 1, 1, AC_MODE_RELU, groups);

  t = ff.conv2d(t, 2 * out_channels, 1, 1, 1, 1, 0, 0, AC_MODE_NONE);

  if ((stride_h > 1 || input->dims[2] != out_channels * 2) && has_residual) {
    input = ff.conv2d(
        input, 2 * out_channels, 1, 1, stride_h, stride_w, 0, 0, AC_MODE_RELU);
    t = ff.relu(ff.add(input, t), false);
  }
  return t;
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffConfig;
  /* { */
  /*   const InputArgs &command_args = HighLevelRuntime::get_input_args(); */
  /*   char **argv = command_args.argv; */
  /*   int argc = command_args.argc; */
  /*   parse_input_args(argv, argc, resnetConfig); */
  log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
                ffConfig.batchSize,
                ffConfig.workersPerNode,
                ffConfig.numNodes);
  /* } */
  FFModel ff(ffConfig);

  Tensor input;
  {
    int const dims[] = {ffConfig.batchSize, 3, 224, 224};
    input = ff.create_tensor<4>(dims, DT_FLOAT);
  }

  Tensor t = input;
  t = ff.conv2d(t, 64, 7, 7, 2, 2, 3, 3, AC_MODE_RELU);
  t = ff.pool2d(t, 3, 3, 2, 2, 1, 1, POOL_MAX);

  int stride;

  stride = 1;
  for (int i = 0; i < 3; i++) {
    t = resnext_block(ff, t, stride, stride, 128, 32);
  }
  stride = 2;
  for (int i = 0; i < 4; i++) {
    t = resnext_block(ff, t, stride, stride, 256, 32);
    stride = 1;
  }
  stride = 2;
  for (int i = 0; i < 6; i++) {
    t = resnext_block(ff, t, stride, stride, 512, 32);
    stride = 1;
  }
  stride = 2;
  for (int i = 0; i < 3; i++) {
    t = resnext_block(ff, t, stride, stride, 1024, 32);
    stride = 1;
  }

  t = ff.relu(t, false);
  t = ff.pool2d(t, t->dims[0], t->dims[1], 1, 1, 0, 0, POOL_AVG);
  t = ff.flat(t);
  t = ff.dense(t, 1000 /*1000*/);
  t = ff.softmax(t);

  Optimizer *optimizer = new SGDOptimizer(&ff, 0.001f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);
  // Data Loader
  /* DataLoader data_loader(ff, resnetConfig, input, ff.label_tensor); */
  ff.init_operators();
  // Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    /* data_loader.reset(); */
    ff.reset_metrics();
    int iterations = 128; // data_loader.num_samples / ffConfig.batchSize;

    for (int iter = 0; iter < iterations; iter++) {
      /* if (resnetConfig.dataset_path.length() == 0) { */
      /*   // Only load data once for random input */
      /*   if (iter == 0 && epoch == 0) */
      /*     data_loader.next_batch(ff); */
      /* } else { */
      /*   data_loader.next_batch(ff); */
      /* } */
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
