#include "benchmark/substitutions/pcg_pattern.h"
#include "models/yolov10/yolov10.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/pcg_from_computation_graph.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/benchmark_utils/loop.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

static ComputationGraph
    make_dense_relu_chain_graph(nonnegative_int num_layers) {
  ComputationGraphBuilder cgb;

  positive_int batch_size = 8_p;
  positive_int num_channels = 128_p;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<positive_int>{
          batch_size,
          num_channels,
      }},
      DataType::FLOAT,
  };

  tensor_guid_t t = cgb.create_input(input_shape, CreateGrad::YES);

  for (nonnegative_int _ : nonnegative_range(num_layers)) {
    t = cgb.dense(
        /*input=*/t,
        /*outDim=*/num_channels,
        /*activation=*/std::nullopt,
        /*use_bias=*/false);
    t = cgb.relu(t);
  }

  return cgb.computation_graph;
}

void benchmark_find_pattern_matches_small(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  ComputationGraph cg = make_dense_relu_chain_graph(2_n);

  ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

  SubParallelComputationGraph subpcg = sub_pcg_from_full_pcg(pcg);

  Substitution substitution = create_fuse_linear_activation(Activation::RELU);

  LOOP(1, dry_run) {
    std::vector<PCGPatternMatch> pattern_matches =
        find_pattern_matches(substitution.pcg_pattern, subpcg);
  }
}

void benchmark_find_pattern_matches_medium(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  ComputationGraph cg = make_dense_relu_chain_graph(16_n);

  ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

  SubParallelComputationGraph subpcg = sub_pcg_from_full_pcg(pcg);

  Substitution substitution = create_fuse_linear_activation(Activation::RELU);

  LOOP(1, dry_run) {
    std::vector<PCGPatternMatch> pattern_matches =
        find_pattern_matches(substitution.pcg_pattern, subpcg);
  }
}

void benchmark_find_pattern_matches_large(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  YOLOv10Config config = get_yolov10_config(
      /*scale=*/YOLOv10Scale::EXTRA_LARGE,
      /*batch_size=*/8_p,
      /*end2end=*/false);

  ComputationGraph cg = get_yolov10_computation_graph(config);

  ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

  SubParallelComputationGraph subpcg = sub_pcg_from_full_pcg(pcg);

  Substitution substitution =
      create_fuse_batch_norm_activation(Activation::SILU);

  LOOP(1, dry_run) {
    std::vector<PCGPatternMatch> pattern_matches =
        find_pattern_matches(substitution.pcg_pattern, subpcg);
  }
}

} // namespace FlexFlow
