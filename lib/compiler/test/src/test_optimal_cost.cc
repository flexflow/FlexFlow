#include "compiler/unity_algorithm.h"
#include "doctest/doctest.h"
#include "test_cost_estimator.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("optimal_cost_0") {
    ParallelComputationGraphBuilder builder;

    ParallelTensorShape input_shape =
        ParallelTensorShape{ParallelTensorDims{
                                FFOrdered<ShardParallelDim>{
                                    ShardParallelDim{32, 2},
                                    ShardParallelDim{16, 1},
                                },
                                ReplicaParallelDimSet{
                                    SumDegree{1},
                                    DiscardCopyDegree{1},
                                },
                            },
                            DataType::FLOAT};

    parallel_tensor_guid_t input0 =
        builder.create_input_tensor(input_shape, true, "input0");
    parallel_tensor_guid_t dense0 = builder.dense(input0,
                                                  8,
                                                  Activation::RELU,
                                                  true,
                                                  DataType::FLOAT,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  "dense0");

    ParallelComputationGraph pcg = builder.pcg;

    auto test_allowed_machine_views = [](ParallelLayerAttrs const &,
                                         MachineSpecification const &) {
      return std::unordered_set<MachineView>{
          make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))};
    };

    CostEstimator estimator = CostEstimator::create<TestCostEstimator>();
    MachineSpecification machine_spec{1, 1, 1, 1, 1};
    OptimalCostCache cached_results;
    OptimalCostResult result = optimal_cost(
        pcg,
        test_allowed_machine_views,
        estimator,
        machine_spec,
        cached_results);

    CHECK(bool(result.runtime > 0));
  }
}