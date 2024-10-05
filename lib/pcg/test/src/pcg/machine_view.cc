#include "pcg/machine_view.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/containers/transform.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MachineView - utility functions") {
    MachineView mv = MachineView{
        MachineSpaceCoordinate{
            /*node_idx=*/0, /*device_idx=*/0, DeviceType::GPU},
        {MachineViewDimension{stride_t{2},
                              MachineSpecificationDimension::INTER_NODE},
         MachineViewDimension{stride_t{2},
                              MachineSpecificationDimension::INTER_NODE}}};

    SUBCASE("num_dims") {
      CHECK(num_dims(mv) == 2);
    }
    SUBCASE("get_device_type") {
      CHECK(get_device_type(mv) == DeviceType::GPU);
    }
  }

  TEST_CASE("get_machine_space_coordinate") {
    SUBCASE("1D case") {

      // This operator has shape (3,), and thus 3 tasks.
      // The (only) dimension is projected on the INTER (device) dimension with
      // a stride of 2. The start of the projection defined by MachineView
      // starts at MachineSpaceCoordinate (0,1), and the machine space has 1
      // node and 6 devices per node.

      /**
       * The tasks will thus be distributed like this:
       *  +-------+-------+-------+-------+-------+-------+
       *  |       | (0,)  |       | (1,)  |       | (2,)  |
       *  +-------+-------+-------+-------+-------+-------+
       * Where the (x,) are the `TaskSpaceCoordinate`s, and the underlying grid
       * is the machine space.
       */
      OperatorTaskSpace task = OperatorTaskSpace{{3}};
      MachineView mv = MachineView{
          MachineSpaceCoordinate{
              /*node_idx=*/0, /*device_idx=*/1, DeviceType::GPU},
          {MachineViewDimension{stride_t{2},
                                MachineSpecificationDimension::INTRA_NODE}}};
      MachineSpecification ms =
          MachineSpecification{/*num_nodes=*/1,
                               /*num_cpus_per_node=*/6,
                               /*num_gpus_per_node=*/6,
                               /*inter_node_bandwidth=*/0,
                               /*intra_node_bandwidth=*/0};

      SUBCASE("Task with TaskSpaceCoordinate = (0,)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0}};
        MachineSpaceCoordinate correct = MachineSpaceCoordinate{
            /*node_idx=*/0, /*device_idx=*/1, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms).value();
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (1,)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1}};
        MachineSpaceCoordinate correct = MachineSpaceCoordinate{
            /*node_idx=*/0, /*device_idx=*/3, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms).value();
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (2,)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{2}};
        MachineSpaceCoordinate correct = MachineSpaceCoordinate{
            /*node_idx=*/0, /*device_idx=*/5, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms).value();
        CHECK(correct == result);
      }

      SUBCASE("TaskSpaceCoordinate is out of bounds") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{4}};
        std::optional<MachineSpaceCoordinate> result =
            get_machine_space_coordinate(task, mv, coord, ms);
        std::optional<MachineSpaceCoordinate> correct = std::nullopt;
        CHECK(result == correct);
      }

      SUBCASE("2D case - projection on different dimensions") {
        // This operator has shape (2, 2), and thus 2 * 2 = 4 tasks.
        // The first dimension is projected onto the INTER (node) dimension with
        // stride 1, while the second dimension is projected onto the INTRA
        // (device) dimension with stride 2. The start of the projection defined
        // by MachineView is at MachineSpaceCoordinates (1, 2), and the machine
        // space has 3 nodes and 5 devices per node.

        /**
         * The tasks will thus be distributed like this:
         *  +-------+-------+-------+-------+-------+
         *  |       |       |       |       |       |
         *  +-------+-------+-------+-------+-------+
         *  |       |       | (0,0) |       | (0,1) |
         *  +-------+-------+-------+-------+-------+
         *  |       |       | (1,0) |       | (1,1) |
         *  +-------+-------+-------+-------+-------+
         * Where the (x,y) are the `TaskSpaceCoordinate`s, and the underlying
         * grid is the machine space.
         */

        OperatorTaskSpace task = OperatorTaskSpace{{2, 2}};
        MachineView mv = MachineView{
            MachineSpaceCoordinate{
                /*node_idx=*/1, /*device_idx=*/2, DeviceType::GPU},
            {MachineViewDimension{stride_t{1},
                                  MachineSpecificationDimension::INTER_NODE},
             MachineViewDimension{stride_t{2},
                                  MachineSpecificationDimension::INTRA_NODE}}};
        MachineSpecification ms =
            MachineSpecification{/*num_nodes=*/3,
                                 /*num_cpus_per_node=*/5,
                                 /*num_gpus_per_node=*/5,
                                 /*inter_node_bandwidth=*/0,
                                 /*intra_node_bandwidth=*/0};

        SUBCASE("Task with TaskSpaceCoordinate = (0,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0, 0}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1, /*device_idx=*/2, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (0,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0, 1}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1, /*device_idx=*/4, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1, 0}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/2, /*device_idx=*/2, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1, 1}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/2, /*device_idx=*/4, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }
      }

      SUBCASE("2D case - projection on same dimension") {
        // This operator has shape (2, 2), and thus 2 * 2 = 4 tasks.
        // Both dimensions are projected on the INTRA (device) dimension, with
        // strides 1 and 2 respectively. The start of the projection defined by
        // MachineView is at MachineSpaceCoordinates (1, 0), and the machine
        // space has 2 nodes and 6 devices per node.

        /**
         *  +-------+-------+-------+-------+-------+-------+
         *  | (0,0) | (1,0) |       |       | (0,1) | (1,1) |
         *  +-------+-------+-------+-------+-------+-------+
         * Where the (x,y) are the `TaskSpaceCoordinate`s, and the underlying
         * grid is the machine space.
         */

        OperatorTaskSpace task = OperatorTaskSpace{{2, 2}};
        MachineView mv = MachineView{
            MachineSpaceCoordinate{
                /*node_idx=*/1, /*device_idx=*/0, DeviceType::GPU},
            {MachineViewDimension{stride_t{1},
                                  MachineSpecificationDimension::INTRA_NODE},
             MachineViewDimension{stride_t{2},
                                  MachineSpecificationDimension::INTRA_NODE}}};
        MachineSpecification ms =
            MachineSpecification{/*num_nodes=*/2,
                                 /*num_cpus_per_node=*/6,
                                 /*num_gpus_per_node=*/6,
                                 /*inter_node_bandwidth=*/0,
                                 /*intra_node_bandwidth=*/0};

        SUBCASE("Task with TaskSpaceCoordinate = (0,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0, 0}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1, /*device_idx=*/0, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (0,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0, 1}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1, /*device_idx=*/4, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1, 0}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1, /*device_idx=*/1, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1, 1}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1, /*device_idx=*/5, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }
      }

      SUBCASE("3D case") {
        // This operator has shape (2, 2, 2), and thus 2 * 2 * 2 = 8 tasks.
        // - The first dimension is projected onto the INTER (node) dimension
        // with stride 1,
        // - The second dimension is projected onto the INTRA (device) dimension
        // with stride 2,
        // - The third dimension is projected onto the INTRA (device) dimension
        // with stride 1. The start of the projection defined by MachineView is
        // at MachineSpaceCoordinates (0, 1), and the machine space has 2 nodes
        // and 8 devices per node.

        /**
         * The tasks will thus be distributed like this:
         *  +-------+-------+-------+-------+-------+-------+-------+-------+
         *  |       | (0,0,0) |       | (0,0,1) |       | (0,1,0) |       |
         * (0,1,1) |
         *  +-------+-------+-------+-------+-------+-------+-------+-------+
         *  |       | (1,0,0) |       | (1,0,1) |       | (1,1,0) |       |
         * (1,1,1) |
         *  +-------+-------+-------+-------+-------+-------+-------+-------+
         * Where the (x,y,z) are the `TaskSpaceCoordinate`s, and the underlying
         * grid is the machine space.
         */

        OperatorTaskSpace task = OperatorTaskSpace{{2, 2, 2}};
        MachineView mv = MachineView{
            MachineSpaceCoordinate{
                /*node_idx=*/0, /*device_idx=*/1, DeviceType::GPU},
            {MachineViewDimension{stride_t{1},
                                  MachineSpecificationDimension::INTER_NODE},
             MachineViewDimension{stride_t{2},
                                  MachineSpecificationDimension::INTRA_NODE},
             MachineViewDimension{stride_t{1},
                                  MachineSpecificationDimension::INTRA_NODE}}};
        MachineSpecification ms =
            MachineSpecification{/*num_nodes=*/2,
                                 /*num_cpus_per_node=*/8,
                                 /*num_gpus_per_node=*/8,
                                 /*inter_node_bandwidth=*/0,
                                 /*intra_node_bandwidth=*/0};

        SUBCASE("Task with TaskSpaceCoordinate = (0,0,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0, 1, 0}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/0, /*device_idx=*/3, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,1,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1, 0, 1}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1, /*device_idx=*/5, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,1,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1, 1, 1}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1, /*device_idx=*/7, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }
      }
    }
  }
}
