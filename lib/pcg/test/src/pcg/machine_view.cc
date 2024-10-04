#include "pcg/machine_view.h"
#include "utils/containers/transform.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("MachineView - utility functions") {
    MachineView mv = MachineView{{stride_t{2}, stride_t{2}},
                                 {MachineSpecificationDimension::INTER_NODE,
                                  MachineSpecificationDimension::INTER_NODE},
                                 MachineSpaceCoordinate{0, 0, DeviceType::GPU}};

    SUBCASE("num_dims") {
      CHECK(num_dims(mv) == 2);
    }
    SUBCASE("get_device_type") {
      CHECK(get_device_type(mv) == DeviceType::GPU);
    }
  }

  TEST_CASE("get_machine_space_coordinate") {
    SUBCASE("1D case") {

      TaskSpaceOperator task = TaskSpaceOperator{{num_points_t{3}}};
      MachineView mv =
          MachineView{{{stride_t{2}}},
                      {{MachineSpecificationDimension::INTRA_NODE}},
                      MachineSpaceCoordinate{0, 1, DeviceType::GPU}};

      MachineSpecification ms = MachineSpecification{
          1, 6, 6, 0, 0}; // Single node with 6 GPUs (0,1,2,3,4,5)

      SUBCASE("Fragment 0") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0}};
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{0, 1, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms);
        CHECK(correct == result);
      }

      SUBCASE("Fragment 1") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1}};
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{0, 3, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms);
        CHECK(correct == result);
      }

      SUBCASE("Fragment 2") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{2}};
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{0, 5, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms);
        CHECK(correct == result);
      }
    }
    SUBCASE("2D case") {

      TaskSpaceOperator task =
          TaskSpaceOperator{{num_points_t{2}, num_points_t{2}}};
      MachineView mv =
          MachineView{{{stride_t{1}, stride_t{2}}},
                      {{MachineSpecificationDimension::INTER_NODE,
                        MachineSpecificationDimension::INTRA_NODE}},
                      MachineSpaceCoordinate{1, 2, DeviceType::GPU}};

      MachineSpecification ms =
          MachineSpecification{3, 5, 5, 0, 0}; // 3 Nodes, 5 GPUs each

      SUBCASE("Fragment (0,0)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0, 0}};
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{1, 2, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms);
        CHECK(correct == result);
      }

      SUBCASE("Fragment (0,1)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0, 1}};
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{1, 4, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms);
        CHECK(correct == result);
      }
      SUBCASE("Fragment (1,0)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1, 0}};
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{2, 2, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms);
        CHECK(correct == result);
      }

      SUBCASE("Fragment (1,1)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1, 1}};
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{2, 4, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms);
        CHECK(correct == result);
      }
    }

    SUBCASE("3D case") {
      TaskSpaceOperator task = TaskSpaceOperator{
          {num_points_t{2}, num_points_t{2}, num_points_t{2}}};
      MachineView mv =
          MachineView{{{stride_t{1}, stride_t{2}, stride_t{1}}},
                      {{MachineSpecificationDimension::INTER_NODE,
                        MachineSpecificationDimension::INTRA_NODE,
                        MachineSpecificationDimension::INTRA_NODE}},
                      MachineSpaceCoordinate{0, 1, DeviceType::GPU}};

      MachineSpecification ms =
          MachineSpecification{2, 8, 8, 0, 0}; // 2 Nodes, 8 GPUs each

      SUBCASE("Fragment (0,0,1)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0, 1, 0}};
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate(0, 3, DeviceType::GPU);
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms);
        ;
        CHECK(correct == result);
      }

      SUBCASE("Fragment (1, 1, 0)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1, 0, 1}};
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{1, 5, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms);
        ;
        CHECK(correct == result);
      }
    }
  }
}
