#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "./cost_estimator_for_test.h"
#include <doctest/doctest.h>
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/get_only.h"

using namespace FlexFlow;


TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_optimal_machine_mapping") {
    auto allowed_machine_views1 = [&](ParallelLayerAttrs const &,
                                      MachineSpecification const &) {
      return std::unordered_set<MachineView>{
          make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))};
    };
    MachineSpecification machine_spec = MachineSpecification{
      /*num_nodes=*/2, 
      /*num_cpus_per_node=*/1, 
      /*num_gpus_per_node=*/1, 
      /*inter_node_bandwidth=*/1, 
      /*intra_node_bandwidth=*/1,
    };

    CostEstimator cost_estimator = make_fake_cost_estimator(
      std::unordered_map<OpCostEstimateKey, float>{}, 
      std::unordered_map<TensorSetMovement, float>{});

    SUBCASE("single layer") {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      MachineView mv1 = make_1d_machine_view(gpu_id_t{1}, gpu_id_t{2});

      auto allowed_machine_views = [&](ParallelLayerAttrs const &,
                                        MachineSpecification const &) {
        return std::unordered_set<MachineView>{mv1};
      };

      ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
        PCGOperatorAttrs{
          InputAttrs{},
        },
        std::nullopt,
      };

      ParallelTensorAttrs output_tensor_attrs = ParallelTensorAttrs{
        ParallelTensorShape{
          ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
              ShardParallelDim{10, 1},
            },
            ReplicaParallelDimSet{
              SumDegree{1},
              DiscardCopyDegree{1},
            },
          },
          DataType::FLOAT,
        },
        /*sync_type=*/std::nullopt,
        /*initializer=*/std::nullopt,
        /*create_gradients=*/CreateGrad::YES,
      };

      ParallelLayerAddedResult added = add_parallel_layer(pcg, 
                                                          layer_attrs,
                                                          {},
                                                          {output_tensor_attrs});
      parallel_layer_guid_t layer = added.parallel_layer;
      parallel_tensor_guid_t output_tensor = get_only(added.outputs);

      MachineMappingCache cache;

      MachineMappingResult result = get_optimal_machine_mapping(pcg, 
                                                                allowed_machine_views, 
                                                                cost_estimator, 
                                                                machine_spec, 
                                                                cache);
      MachineMappingResult correct = MachineMappingResult{
        /*runtime=*/2.0,
        /*machine_mapping=*/MachineMapping{{
          {layer, mv1},
        }},
      };

      CHECK(result == correct);
    }

    SUBCASE("pair of layers in sequence") {
      FAIL("TODO");
    }

    SUBCASE("pair of layers in parallel") {
      FAIL("TODO");
    }

    // SUBCASE("simple PCG") {
    //
    //   ParallelComputationGraph pcg_simple = [&] {
    //     ParallelComputationGraphBuilder builder;
    //
    //     ParallelTensorShape input_shape0 =
    //         ParallelTensorShape{ParallelTensorDims{
    //                                 FFOrdered<ShardParallelDim>{
    //                                     ShardParallelDim{32, 2},
    //                                     ShardParallelDim{32, 1},
    //                                     ShardParallelDim{16, 1},
    //                                 },
    //                                 ReplicaParallelDimSet{
    //                                     SumDegree{1},
    //                                     DiscardCopyDegree{1},
    //                                 },
    //                             },
    //                             DataType::FLOAT};
    //
    //     ParallelTensorShape input_shape1 =
    //         ParallelTensorShape{ParallelTensorDims{
    //                                 FFOrdered<ShardParallelDim>{
    //                                     ShardParallelDim{32, 2},
    //                                     ShardParallelDim{16, 1},
    //                                     ShardParallelDim{8, 1},
    //                                 },
    //                                 ReplicaParallelDimSet{
    //                                     SumDegree{1},
    //                                     DiscardCopyDegree{1},
    //                                 },
    //                             },
    //                             DataType::FLOAT};
    //
    //     parallel_tensor_guid_t input0 =
    //         builder.create_input_tensor(input_shape0);
    //     parallel_tensor_guid_t input1 =
    //         builder.create_input_tensor(input_shape1);
    //     parallel_tensor_guid_t dense0 = builder.batch_matmul(input0, input1);
    //
    //     return builder.pcg;
    //   }();
    //
    //   MachineMappingResult result =
    //       get_optimal_machine_mapping(pcg_simple,
    //                                   allowed_machine_views1,
    //                                   estimator1,
    //                                   machine_spec1,
    //                                   cached_results1);
    //
    //   CHECK(result.runtime == 3);
    // }

    // SUBCASE("PCG is a chain") {
    //   ParallelComputationGraph pcg_chain = [&] {
    //     ParallelComputationGraphBuilder builder;
    //
    //     ParallelTensorShape input_shape =
    //         ParallelTensorShape{ParallelTensorDims{
    //                                 FFOrdered<ShardParallelDim>{
    //                                     ShardParallelDim{32, 2},
    //                                     ShardParallelDim{16, 1},
    //                                 },
    //                                 ReplicaParallelDimSet{
    //                                     SumDegree{1},
    //                                     DiscardCopyDegree{1},
    //                                 },
    //                             },
    //                             DataType::FLOAT};
    //
    //     parallel_tensor_guid_t input0 =
    //         builder.create_input_tensor(input_shape);
    //     parallel_tensor_guid_t layer1 = builder.identity(input0);
    //     parallel_tensor_guid_t layer2 = builder.identity(layer1);
    //     parallel_tensor_guid_t layer3 = builder.identity(layer2);
    //     parallel_tensor_guid_t layer4 = builder.identity(layer3);
    //     parallel_tensor_guid_t layer5 = builder.identity(layer4);
    //     parallel_tensor_guid_t layer6 = builder.identity(layer5);
    //
    //     return builder.pcg;
    //   }();
    //
    //   MachineMappingResult result =
    //       get_optimal_machine_mapping(pcg_chain,
    //                                   allowed_machine_views1,
    //                                   estimator1,
    //                                   machine_spec1,
    //                                   cached_results1);
    //   CHECK(result.runtime == 13);
    // }
    //
    // SUBCASE("PCG has multiple chains") {
    //   ParallelComputationGraph pcg_multiple_chains = [&] {
    //     ParallelComputationGraphBuilder builder;
    //
    //     ParallelTensorShape input_shape0 =
    //         ParallelTensorShape{ParallelTensorDims{
    //                                 FFOrdered<ShardParallelDim>{
    //                                     ShardParallelDim{32, 2},
    //                                     ShardParallelDim{32, 1},
    //                                     ShardParallelDim{16, 1},
    //                                 },
    //                                 ReplicaParallelDimSet{
    //                                     SumDegree{1},
    //                                     DiscardCopyDegree{1},
    //                                 },
    //                             },
    //                             DataType::FLOAT};
    //
    //     ParallelTensorShape input_shape1 =
    //         ParallelTensorShape{ParallelTensorDims{
    //                                 FFOrdered<ShardParallelDim>{
    //                                     ShardParallelDim{32, 2},
    //                                     ShardParallelDim{16, 1},
    //                                     ShardParallelDim{8, 1},
    //                                 },
    //                                 ReplicaParallelDimSet{
    //                                     SumDegree{1},
    //                                     DiscardCopyDegree{1},
    //                                 },
    //                             },
    //                             DataType::FLOAT};
    //
    //     parallel_tensor_guid_t input0 =
    //         builder.create_input_tensor(input_shape0);
    //     parallel_tensor_guid_t input1 =
    //         builder.create_input_tensor(input_shape1);
    //     parallel_tensor_guid_t relu0 = builder.relu(input0);
    //     parallel_tensor_guid_t relu1 = builder.relu(input1);
    //     parallel_tensor_guid_t matmul0 = builder.batch_matmul(relu0, relu1);
    //
    //     return builder.pcg;
    //   }();
    //
    //   MachineMappingResult result =
    //       get_optimal_machine_mapping(pcg_multiple_chains,
    //                                   allowed_machine_views1,
    //                                   estimator1,
    //                                   machine_spec1,
    //                                   cached_results1);
    //   CHECK(result.runtime == 5);
    // }
    //
    // SUBCASE("PCG is not sp-izable due to multiple inputs") {
    //   ParallelComputationGraph pcg_non_sp = [&] {
    //     ParallelComputationGraphBuilder builder;
    //
    //     ParallelTensorShape input_shape =
    //         ParallelTensorShape{ParallelTensorDims{
    //                                 FFOrdered<ShardParallelDim>{
    //                                     ShardParallelDim{32, 2},
    //                                     ShardParallelDim{16, 1},
    //                                 },
    //                                 ReplicaParallelDimSet{
    //                                     SumDegree{1},
    //                                     DiscardCopyDegree{1},
    //                                 },
    //                             },
    //                             DataType::FLOAT};
    //
    //     parallel_tensor_guid_t input0 =
    //         builder.create_input_tensor(input_shape);
    //     parallel_tensor_guid_t dense0 = builder.dense(input0, 8);
    //     parallel_tensor_guid_t dense1 = builder.dense(input0, 4);
    //     parallel_tensor_guid_t dense2 = builder.dense(dense1, 8);
    //     parallel_tensor_guid_t add0 = builder.add(dense0, dense2);
    //
    //     return builder.pcg;
    //   }();
    //
    //   // TODO: Handle this case in compiler
    //   // TODO: separate testcases for this too that actually check the graph
    //   // manipulation
    //   if (false) {
    //     MachineMappingResult result =
    //         get_optimal_machine_mapping(pcg_non_sp,
    //                                     allowed_machine_views1,
    //                                     estimator1,
    //                                     machine_spec1,
    //                                     cached_results1);
    //     CHECK(bool(result.runtime > 0));
    //     CHECK(result.runtime == 7);
    //   }
    // }
  }
}
