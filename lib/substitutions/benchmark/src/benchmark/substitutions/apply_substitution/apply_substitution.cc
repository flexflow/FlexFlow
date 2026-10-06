#include "benchmark/substitutions/apply_substitution/apply_substitution.h"
#include "models/yolov10/yolov10.h"
#include "pcg/pcg_from_computation_graph.h"
#include "substitutions/apply_substitution/apply_substitution.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/benchmark_utils/loop.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

void benchmark_apply_substitution(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  ComputationGraph cg = []() {
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

    nonnegative_int num_layers = 8_n;

    for (nonnegative_int _ : nonnegative_range(num_layers)) {
      t = cgb.dense(
          /*input=*/t,
          /*outDim=*/num_channels,
          /*activation=*/std::nullopt,
          /*use_bias=*/false);
      t = cgb.relu(t);
    }

    return cgb.computation_graph;
  }();

  ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

  SubParallelComputationGraph subpcg = sub_pcg_from_full_pcg(pcg);

  Substitution substitution = create_fuse_linear_activation(Activation::RELU);

  std::vector<PCGPatternMatch> pattern_matches =
      find_pattern_matches(substitution.pcg_pattern, subpcg);

  LOOP(1, dry_run) {
    for (PCGPatternMatch const &match : pattern_matches) {
      apply_substitution(subpcg, substitution, match);
    }
  }
}

} // namespace FlexFlow
