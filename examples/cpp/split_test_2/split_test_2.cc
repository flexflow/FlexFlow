#include "flexflow/model.h"
#include "flexflow/substitution.h"

namespace FlexFlow {

using namespace Legion;
using namespace FlexFlow;
using FlexFlow::PCG::Graph;
using FlexFlow::PCG::GraphSearchHelper;
using FlexFlow::PCG::Node;

Legion::Logger log_app("split_test_2");

void top_level_task(Task const *task,
                    std::vector<PhysicalRegion> const &regions,
                    Context ctx,
                    Runtime *runtime) {
  FFConfig ffConfig;
  log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
                ffConfig.batchSize,
                ffConfig.workersPerNode,
                ffConfig.numNodes);
  FFModel ff(ffConfig);

  Tensor input;
  {
    int const dims[] = {ffConfig.batchSize, 4, 32, 32};
    input = ff.create_tensor<4>(dims, DT_FLOAT);
    log_app.print(
        "input size: %d %d %d %d", dims[0], dims[1], dims[2], dims[3]);
  }

  Tensor t, t1, t2;

  // int channels[] = { 4, 8, 16, 32, 64, 128, 256, 512  };
  int channels[] = {4, 8, 16};

  t = input;
  for (int i = 0; i < sizeof(channels) / sizeof(int); i++) {
    t = ff.conv2d(t, channels[1], 3, 3, 2, 2, 0, 0);
    std::ostringstream oss;
    oss << "Iteration " << i;
    t->print(oss.str());
  }
  t->print("Post-conv shape");
  t = ff.flat(t);
  t = ff.relu(t);
  t = ff.softmax(t);

  Optimizer *optimizer = new SGDOptimizer(&ff, 0.001f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);
  GraphSearchHelper gsh(&ff);
  std::unique_ptr<Graph> best_graph;
  std::unordered_map<Node, MachineView> optimal_views;
  gsh.graph_optimize(10, false, best_graph, optimal_views);
  // {
  // runtime->issue_execution_fence(ctx);
  // TimingLauncher timer(MEASURE_MICRO_SECONDS);
  // Future future = runtime->issue_timing_measurement(ctx, timer);
  // future.get_void_result();
  // }
  // double ts_start = Realm::Clock::current_time_in_microseconds();
  // for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
  // ff.reset_metrics();
  // int iterations = 128; // data_loader.num_samples / ffConfig.batchSize;
  //
  // for (int iter = 0; iter < iterations; iter++) {
  // runtime->begin_trace(ctx, 111/*trace_id*/);
  // ff.forward();
  // ff.zero_gradients();
  // ff.backward();
  // ff.update();
  // runtime->end_trace(ctx, 111/*trace_id*/);
  // }
  // }
  // End timer
  // {
  // runtime->issue_execution_fence(ctx);
  // TimingLauncher timer(MEASURE_MICRO_SECONDS);
  // Future future = runtime->issue_timing_measurement(ctx, timer);
  // future.get_void_result();
  // }
  // double ts_end = Realm::Clock::current_time_in_microseconds();
  // double run_time = 1e-6 * (ts_end - ts_start);
  // printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n", run_time,
  //  128 * ffConfig.batchSize * ffConfig.epochs / run_time);
}

void register_custom_tasks() {}

} // namespace FlexFlow
