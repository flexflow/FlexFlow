#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "op-attrs/initializer_attrs.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("mapped_pcgs_are_isomorphic") {
    auto make_mpcg = []() -> MappedParallelComputationGraph {
      TensorShape input_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  8_p,
                  3_p,
                  6_p,
              },
          },
          DataType::FLOAT,
      };

      ParallelComputationGraphBuilder b;

      std::string input1_name = "input1";
      std::string input2_name = "input2";
      std::string partition1_name = "partition1";
      std::string partition2_name = "partition2";
      std::string add_name = "add";

      parallel_tensor_guid_t t1 =
          b.create_input_tensor(input_shape, input1_name);
      t1 = b.parallel_partition(t1, ff_dim_t{0_n}, 2_ge2, partition1_name);
      parallel_tensor_guid_t t2 =
          b.create_input_tensor(input_shape, input2_name);
      t2 = b.parallel_partition(t2, ff_dim_t{0_n}, 2_ge2, partition2_name);

      parallel_tensor_guid_t t3 = b.add(t1, t2, add_name);

      ParallelComputationGraph pcg = b.pcg;

      parallel_layer_guid_t l_input1 =
          get_parallel_layer_by_name(pcg, input1_name);
      parallel_layer_guid_t l_input2 =
          get_parallel_layer_by_name(pcg, input2_name);
      parallel_layer_guid_t l_partition1 =
          get_parallel_layer_by_name(pcg, partition1_name);
      parallel_layer_guid_t l_partition2 =
          get_parallel_layer_by_name(pcg, partition2_name);
      parallel_layer_guid_t l_add = get_parallel_layer_by_name(pcg, add_name);

      auto machine_coord = [](nonnegative_int x) -> MachineSpaceCoordinate {
        return MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/x,
        };
      };

      auto ptensor_coord =
          [](nonnegative_int x) -> ParallelTensorSpaceCoordinate {
        return ParallelTensorSpaceCoordinate{
            /*sum_component=*/0_n,
            /*discard_copy_component=*/0_n,
            /*shard_components=*/FFOrdered{x, 0_n, 0_n},
        };
      };

      MappedOperatorTaskGroup input_mapping = MappedOperatorTaskGroup{
          bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
              {machine_coord(0_n),
               OperatorAtomicTaskShardBinding{
                   {
                       {TensorSlotName::OUTPUT, ptensor_coord(0_n)},
                   },
               }},
          },
      };

      MappedOperatorTaskGroup partition_mapping = MappedOperatorTaskGroup{
          bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
              {
                  machine_coord(0_n),
                  OperatorAtomicTaskShardBinding{
                      {
                          {TensorSlotName::INPUT, ptensor_coord(0_n)},
                          {TensorSlotName::OUTPUT, ptensor_coord(0_n)},
                      },
                  },
              },
              {
                  machine_coord(1_n),
                  OperatorAtomicTaskShardBinding{
                      {
                          {TensorSlotName::INPUT, ptensor_coord(0_n)},
                          {TensorSlotName::OUTPUT, ptensor_coord(1_n)},
                      },
                  },
              },
          },
      };

      std::map<parallel_layer_guid_t, MappedOperatorTaskGroup> mapped_tasks = {
          {
              l_input1,
              input_mapping,
          },
          {
              l_input2,
              input_mapping,
          },
          {
              l_partition1,
              partition_mapping,
          },
          {
              l_partition2,
              partition_mapping,
          },
          {l_add,
           MappedOperatorTaskGroup{
               bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                   {
                       machine_coord(0_n),
                       OperatorAtomicTaskShardBinding{
                           {
                               {TensorSlotName::LHS_INPUT, ptensor_coord(0_n)},
                               {TensorSlotName::RHS_INPUT, ptensor_coord(0_n)},
                               {TensorSlotName::OUTPUT, ptensor_coord(0_n)},
                           },
                       },
                   },
                   {
                       machine_coord(1_n),
                       OperatorAtomicTaskShardBinding{
                           {
                               {TensorSlotName::LHS_INPUT, ptensor_coord(1_n)},
                               {TensorSlotName::RHS_INPUT, ptensor_coord(1_n)},
                               {TensorSlotName::OUTPUT, ptensor_coord(1_n)},
                           },
                       },
                   },
               },
           }},
      };

      return mapped_pcg_from_pcg_and_mapped_op_task_groups(pcg, mapped_tasks);
    };

    MappedParallelComputationGraph mpcg1 = make_mpcg();
    MappedParallelComputationGraph mpcg2 = make_mpcg();

    CHECK(mapped_pcgs_are_isomorphic(mpcg1, mpcg2));
  }
}
