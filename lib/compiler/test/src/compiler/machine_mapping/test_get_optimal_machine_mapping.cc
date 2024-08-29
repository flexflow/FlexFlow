#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "doctest/doctest.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "cost_estimator_for_test.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_optimal_machine_mapping") {

    ParallelComputationGraph pcg_simple = [&] {
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
          builder.create_input_tensor(input_shape);
      parallel_tensor_guid_t dense0 = builder.dense(input0,
                                                    8);

      return builder.pcg;
    }();

    ParallelComputationGraph pcg_chain = [&] {
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
          builder.create_input_tensor(input_shape);
      parallel_tensor_guid_t dense0 = builder.dense(input0, 8);
      parallel_tensor_guid_t dense1 = builder.dense(dense0, 8);
      parallel_tensor_guid_t dense2 = builder.dense(dense1, 8);
      parallel_tensor_guid_t dense3 = builder.dense(dense2, 8);
      parallel_tensor_guid_t dense4 = builder.dense(dense3, 8);
      parallel_tensor_guid_t dense5 = builder.dense(dense4, 8);

      return builder.pcg;
    }();

    ParallelComputationGraph pcg_multiple_chains = [&] {
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

    ParallelComputationGraph pcg_non_sp = [&] {
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
          builder.create_input_tensor(input_shape);
      parallel_tensor_guid_t dense0 = builder.dense(input0, 8);
      parallel_tensor_guid_t dense1 = builder.dense(input0, 4);
      parallel_tensor_guid_t dense2 = builder.dense(dense1, 8);
      parallel_tensor_guid_t add0 = builder.add(dense0, dense2);

      return builder.pcg;
    }();

    auto allowed_machine_views1 = [&](ParallelLayerAttrs const &,
                                        MachineSpecification const &) {
      // TODO(@Mengdi Wu): Replace it with actual allowed machine views when https://github.com/flexflow/FlexFlow/pull/1458 is merged
      return std::unordered_set<MachineView>{
          make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))};
    };
    CostEstimator estimator1 = CostEstimator::create<CostEstimatorForTest>();
    MachineSpecification machine_spec1(1, 1, 1, 1, 1);
    MachineMappingCache cached_results1;

    SUBCASE("simple PCG") {
      MachineMappingResult result =
          get_optimal_machine_mapping(pcg_simple,
                                      allowed_machine_views1,
                                      estimator1,
                                      machine_spec1,
                                      cached_results1);

      CHECK(bool(result.runtime > 0));
      // TODO(@Mengdi Wu): fill it with actual cost
      // CHECK(result.runtime == xx);
    }

    SUBCASE("PCG is a chain") {
      MachineMappingResult result =
          get_optimal_machine_mapping(pcg_chain,
                                      allowed_machine_views1,
                                      estimator1,
                                      machine_spec1,
                                      cached_results1);
      CHECK(bool(result.runtime > 0));
      // CHECK(result.runtime == xx);
    }

    SUBCASE("PCG has multiple chains") {
      MachineMappingResult result =
          get_optimal_machine_mapping(pcg_multiple_chains,
                                      allowed_machine_views1,
                                      estimator1,
                                      machine_spec1,
                                      cached_results1);
      CHECK(bool(result.runtime > 0));
      // CHECK(result.runtime == xx);
    }

    SUBCASE("PCG is not sp-izable due to multiple inputs") {
      // TODO: Handle this case in compiler
      if (false) {
        MachineMappingResult result =
            get_optimal_machine_mapping(pcg_non_sp,
                                        allowed_machine_views1,
                                        estimator1,
                                        machine_spec1,
                                        cached_results1);
        CHECK(bool(result.runtime > 0));
        // CHECK(result.runtime == xx);
      }
    }
    
  }
}
