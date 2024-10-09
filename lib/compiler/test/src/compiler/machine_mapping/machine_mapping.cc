#include "compiler/machine_mapping/machine_mapping.h"
#include "cost_estimator_for_test.h"
#include "doctest/doctest.h"
#include "pcg/machine_view.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("combine_disjoint_mappings(MachineMapping, MachineMappping)") {
    MachineView machine_view_0 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineMapping machine_mapping_0 = MachineMapping({
        {parallel_layer_guid_t{Node{0}}, machine_view_0},
    });
    MachineMapping machine_mapping_1 = MachineMapping({
        {parallel_layer_guid_t{Node{1}}, machine_view_1},
    });
    MachineMapping correct = MachineMapping{{
        {parallel_layer_guid_t{Node{0}}, machine_view_0},
        {parallel_layer_guid_t{Node{1}}, machine_view_1},
    }};
    MachineMapping result =
        combine_disjoint_mappings(machine_mapping_0, machine_mapping_1);
    CHECK(result == correct);
  }

  TEST_CASE("nodes_are_disjoint(MachineMapping, MachineMappping)") {
    MachineView machine_view_0 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineMapping machine_mapping_0 = MachineMapping({
        {parallel_layer_guid_t{Node{0}}, machine_view_0},
    });

    SUBCASE("nodes are disjoint") {
      MachineMapping machine_mapping_1 = MachineMapping({
          {parallel_layer_guid_t{Node{1}}, machine_view_1},
      });

      bool correct = true;
      bool result = nodes_are_disjoint(machine_mapping_0, machine_mapping_1);
      CHECK(result == correct);
    }

    SUBCASE("nodes are not disjoint") {
      MachineMapping machine_mapping_1 = MachineMapping({
          {parallel_layer_guid_t{Node{0}}, machine_view_0},
          {parallel_layer_guid_t{Node{1}}, machine_view_1},
      });
      bool correct = false;
      bool result = nodes_are_disjoint(machine_mapping_0, machine_mapping_1);
      CHECK(result == correct);
    }
  }
}
