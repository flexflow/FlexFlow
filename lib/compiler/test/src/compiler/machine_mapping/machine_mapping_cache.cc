#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "compiler/machine_mapping/split_sp_decomposition.h"
#include "cost_estimator_for_test.h"
#include "doctest/doctest.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/graph/serial_parallel/get_serial_parallel_decomposition.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MachineMappingCache") {
    ParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder builder;

      ParallelTensorShape input_shape0 =
          ParallelTensorShape{ParallelTensorDims{
                                  FFOrdered<ShardParallelDim>{
                                      ShardParallelDim{32, 2},
                                      ShardParallelDim{32, 1},
                                      ShardParallelDim{16, 1},
                                  },
                                  ReplicaParallelDimSet{
                                      SumDegree{1},
                                      DiscardCopyDegree{1},
                                  },
                              },
                              DataType::FLOAT};

      ParallelTensorShape input_shape1 =
          ParallelTensorShape{ParallelTensorDims{
                                  FFOrdered<ShardParallelDim>{
                                      ShardParallelDim{32, 2},
                                      ShardParallelDim{16, 1},
                                      ShardParallelDim{8, 1},
                                  },
                                  ReplicaParallelDimSet{
                                      SumDegree{1},
                                      DiscardCopyDegree{1},
                                  },
                              },
                              DataType::FLOAT};

      parallel_tensor_guid_t input0 = builder.create_input_tensor(input_shape0);
      parallel_tensor_guid_t input1 = builder.create_input_tensor(input_shape1);
      parallel_tensor_guid_t relu0 = builder.relu(input0);
      parallel_tensor_guid_t relu1 = builder.relu(input1);
      parallel_tensor_guid_t matmul0 = builder.batch_matmul(relu0, relu1);

      return builder.pcg;
    }();

    SerialParallelDecomposition subgraph0 =
        get_serial_parallel_decomposition(pcg.raw_graph).value();
    auto [subgraph1, subgraph2] =
        split_sp_decomposition(subgraph0.get<SerialSplit>());

    MachineSpecification machine_spec(1, 1, 1, 1, 1);
    MachineMappingState state0(subgraph0, machine_spec, {});
    MachineMappingState state1(subgraph1, machine_spec, {});
    MachineMappingState state2(subgraph2, machine_spec, {});

    MachineMappingResult result0(
        2,
        MachineMapping(
            std::unordered_map<parallel_layer_guid_t, MachineView>{}));
    MachineMappingResult result1(
        1,
        MachineMapping(
            std::unordered_map<parallel_layer_guid_t, MachineView>{}));
    MachineMappingResult result2(
        1,
        MachineMapping(
            std::unordered_map<parallel_layer_guid_t, MachineView>{}));

    MachineMappingCache cache;

    cache.save(state0, result0);
    CHECK(cache.load(state0).value() == result0);
    CHECK(!cache.load(state1));
    CHECK(!cache.load(state2));

    cache.save(state1, result1);
    CHECK(cache.load(state0).value() == result0);
    CHECK(cache.load(state1).value() == result1);
    CHECK(!cache.load(state2));

    cache.save(state2, result2);
    CHECK(cache.load(state0).value() == result0);
    CHECK(cache.load(state1).value() == result1);
    CHECK(cache.load(state2).value() == result2);
  }
}
