#include "compiler/unity_algorithm/unity_algorithm.h"
#include "compiler/cost_estimator/runtime_only_cost_estimator_from_cost_estimator.h"
#include "internal/cost_estimator_for_test.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/replica_type.dtg.h"
#include "op-attrs/shard_parallel_dim.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "pcg/pcg_from_computation_graph.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("optimize_with_unity_algorithm") {
    ComputationGraph cg = [&] {
      ComputationGraphBuilder b;
      TensorShape input_tensor_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  32_p,
                  64_p,
              },
          },
          DataType::FLOAT,
      };
      tensor_guid_t t = b.create_input(input_tensor_shape, CreateGrad::YES);
      t = b.dense(t,
                  /*outDim=*/16_p,
                  /*activation=*/std::nullopt);
      t = b.gelu(t);
      t = b.dense(t,
                  /*outDim=*/12_p,
                  /*activation=*/std::nullopt,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt);
      t = b.relu(t);
      t = b.dense(t,
                  /*outDim=*/8_p,
                  /*activation=*/Activation::RELU);
      return b.computation_graph;
    }();

    ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

    RuntimeOnlyCostEstimator cost_estimator =
        runtime_only_cost_estimator_from_cost_estimator(
            make_fake_cost_estimator(
                [](OpCostEstimateKey const &k) -> OpCostMetrics {
                  return OpCostMetrics{
                      /*forward_runtime=*/1.0_ms,
                      /*backward_runtime=*/2.0_ms,
                      /*memory=*/1_bytes,
                  };
                },
                [](TensorSetMovement const &) -> milliseconds_t {
                  return 1.0_ms;
                }));

    MachineComputeSpecification full_machine_spec = MachineComputeSpecification{
        /*num_nodes=*/2_p,
        /*num_cpus_per_node=*/1_p,
        /*num_gpus_per_node=*/1_p,
    };

    std::vector<Substitution> substitution_set =
        get_unity_substitution_set(full_machine_spec);

    SUBCASE("do not apply substitution") {
      UnitySearchConfig search_config = UnitySearchConfig{
          /*alpha=*/1.0,
          /*budget=*/0,
          /*max_num_ops=*/100,
      };
      SearchResult result = optimize_with_unity_algorithm(pcg,
                                                          cost_estimator,
                                                          full_machine_spec,
                                                          search_config,
                                                          substitution_set);
      CHECK(pcgs_are_isomorphic(pcg, result.pcg));
    }

    SUBCASE("apply substitution") {
      UnitySearchConfig search_config = UnitySearchConfig{
          /*alpha=*/1.0,
          /*budget=*/1,
          /*max_num_ops=*/100,
      };
      SearchResult result = optimize_with_unity_algorithm(pcg,
                                                          cost_estimator,
                                                          full_machine_spec,
                                                          search_config,
                                                          substitution_set);
    }
  }
}
