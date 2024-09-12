#include "compiler/graph_optimize_state.h"
#include "doctest/doctest.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("GraphOptimizeState::operator==") {
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

    parallel_tensor_guid_t dense1 = builder.dense(dense0,
                                                  4,
                                                  Activation::RELU,
                                                  true,
                                                  DataType::FLOAT,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  "dense1");

    ParallelComputationGraph pcg = builder.pcg;

    // `machine_mapping` is determined by the PCG and the device mapping
    // algorithm, and `runtime` is determined by the PCG and the device mapping,
    // so their values here do not matter.
    std::unordered_map<parallel_layer_guid_t, MachineView> empty_machine_views;
    MachineMapping empty_machine_mapping(empty_machine_views);
    bool result1 =
        GraphOptimizeState(GraphOptimizeResult(pcg, empty_machine_mapping),
                           0) ==
        GraphOptimizeState(GraphOptimizeResult(pcg, empty_machine_mapping), 0);
    bool correct1 = true;
    CHECK(result1 == correct1);

    ParallelComputationGraphBuilder builder_;

    parallel_tensor_guid_t input0_ =
        builder.create_input_tensor(input_shape, true, "input0");
    parallel_tensor_guid_t dense0_ = builder.dense(input0,
                                                   8,
                                                   Activation::RELU,
                                                   true,
                                                   DataType::FLOAT,
                                                   std::nullopt,
                                                   std::nullopt,
                                                   "dense0");

    ParallelComputationGraph pcg_ = builder.pcg;

    bool result2 =
        GraphOptimizeState(GraphOptimizeResult(pcg, empty_machine_mapping),
                           0) ==
        GraphOptimizeState(GraphOptimizeResult(pcg_, empty_machine_mapping), 0);
    bool correct2 = false;
    CHECK(result2 == correct2);
  }
}
