#include "pcg/start_invariant_machine_view.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("StartInvariantMachineView - utility functions") {
    StartInvariantMachineView simv = StartInvariantMachineView{
        {MachineViewDimension{stride_t{2},
                              MachineSpecificationDimension::INTER_NODE},
         MachineViewDimension{stride_t{2},
                              MachineSpecificationDimension::INTER_NODE}},
        DeviceType::GPU};

    SUBCASE("num_dims") {
      int result = num_dims(simv);
      int correct = 2;
      CHECK(result == correct);
    }

    SUBCASE("get_device_type") {
      DeviceType result = get_device_type(simv);
      DeviceType correct = DeviceType::GPU;
      CHECK(result == correct);
    }

    SUBCASE("get_strides") {
      std::vector<stride_t> result = get_strides(simv);
      std::vector<stride_t> correct = {stride_t{2}, stride_t{2}};
      CHECK(result == correct);
    }

    SUBCASE("get_dimensions") {
      std::vector<MachineSpecificationDimension> result = get_dimensions(simv);
      std::vector<MachineSpecificationDimension> correct = {
          MachineSpecificationDimension::INTER_NODE,
          MachineSpecificationDimension::INTER_NODE};
      CHECK(result == correct);
    }
  }

  TEST_CASE("StartInvariantMachineView - conversions") {
    MachineSpaceCoordinate start =
        MachineSpaceCoordinate{1, 2, DeviceType::GPU};
    std::vector<MachineViewDimension> dimensions = {
        MachineViewDimension{stride_t{2},
                             MachineSpecificationDimension::INTER_NODE},
        MachineViewDimension{stride_t{3},
                             MachineSpecificationDimension::INTRA_NODE}};

    MachineView mv = MachineView{start, dimensions};
    StartInvariantMachineView simv =
        StartInvariantMachineView{dimensions, DeviceType::GPU};

    SUBCASE("start_invariant_from_machine_view") {
      StartInvariantMachineView result = start_invariant_from_machine_view(mv);
      StartInvariantMachineView correct = simv;
      CHECK(result == correct);
    }

    SUBCASE("machine_view_from_start_invariant") {
      MachineView result = machine_view_from_start_invariant(simv, start);
      MachineView correct = mv;
      CHECK(result == correct);
    }

    SUBCASE("conversion is invertible") {
      SUBCASE("MachineView -> StartInvariant -> MachineView") {
        MachineView result = machine_view_from_start_invariant(
            start_invariant_from_machine_view(mv), start);
        MachineView correct = mv;
        CHECK(result == correct);
      }

      SUBCASE("StartInvariant -> MachineView -> StartInvariant") {
        StartInvariantMachineView result = start_invariant_from_machine_view(
            machine_view_from_start_invariant(simv, start));
        StartInvariantMachineView correct = simv;
        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("StartInvariantMachineView - get_machine_space_offset") {
    SUBCASE("1D case") {
      // This operator has shape (3,), and thus 3 tasks.
      // The (only) dimension is projected on the INTRA (device) dimension with
      // a stride of 2. The machine space has 1 node and 6 devices per node.
      /**
       * The tasks will thus be distributed like this:
       *  +-------+-------+-------+-------+-------+-------+
       *  | (0,)  |       | (1,)  |       | (2,)  |       |
       *  +-------+-------+-------+-------+-------+-------+
       */
      OperatorTaskSpace task = OperatorTaskSpace{{3}};
      StartInvariantMachineView simv = StartInvariantMachineView{
          {MachineViewDimension{stride_t{2},
                                MachineSpecificationDimension::INTRA_NODE}},
          DeviceType::GPU};
      MachineSpecification ms =
          MachineSpecification{/*num_nodes=*/1,
                               /*num_cpus_per_node=*/6,
                               /*num_gpus_per_node=*/6,
                               /*inter_node_bandwidth=*/0,
                               /*intra_node_bandwidth=*/0};

      SUBCASE("get_machine_space_offset") {
        SUBCASE("Task with TaskSpaceCoordinate = (0,)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0}};
          MachineSpaceOffset correct =
              MachineSpaceOffset{0, 0, DeviceType::GPU};
          MachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1}};
          MachineSpaceOffset correct =
              MachineSpaceOffset{0, 2, DeviceType::GPU};
          MachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (2,)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{2}};
          MachineSpaceOffset correct =
              MachineSpaceOffset{0, 4, DeviceType::GPU};
          MachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord, ms).value();
          CHECK(correct == result);
        }
      }

      SUBCASE("get_machine_space_offsets") {
        std::unordered_set<MachineSpaceOffset> correct = {
            MachineSpaceOffset{0, 0, DeviceType::GPU},
            MachineSpaceOffset{0, 2, DeviceType::GPU},
            MachineSpaceOffset{0, 4, DeviceType::GPU}};
        std::unordered_set<MachineSpaceOffset> result =
            get_machine_space_offsets(task, simv, ms);
        CHECK(correct == result);
      }
    }

    SUBCASE("2D case") {
      // This operator has shape (2, 2), and thus 2 * 2 = 4 tasks.
      // The first dimension is projected onto the INTER (node) dimension with
      // stride 1, while the second dimension is projected onto the INTRA
      // (device) dimension with stride 2. The machine space has 2 nodes and 4
      // devices per node.

      /**
       * The tasks will thus be distributed like this:
       *  +-------+-------+-------+-------+
       *  | (0,0) |       | (0,1) |       |
       *  +-------+-------+-------+-------+
       *  | (1,0) |       | (1,1) |       |
       *  +-------+-------+-------+-------+
       */

      OperatorTaskSpace task = OperatorTaskSpace{{2, 2}};
      StartInvariantMachineView simv = StartInvariantMachineView{
          {MachineViewDimension{stride_t{1},
                                MachineSpecificationDimension::INTER_NODE},
           MachineViewDimension{stride_t{2},
                                MachineSpecificationDimension::INTRA_NODE}},
          DeviceType::GPU};
      MachineSpecification ms =
          MachineSpecification{/*num_nodes=*/2,
                               /*num_cpus_per_node=*/4,
                               /*num_gpus_per_node=*/4,
                               /*inter_node_bandwidth=*/0,
                               /*intra_node_bandwidth=*/0};

      SUBCASE("get_machine_space_offset") {
        SUBCASE("Task with TaskSpaceCoordinate = (0,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0, 0}};
          MachineSpaceOffset correct =
              MachineSpaceOffset{0, 0, DeviceType::GPU};
          MachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (0,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0, 1}};
          MachineSpaceOffset correct =
              MachineSpaceOffset{0, 2, DeviceType::GPU};
          MachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1, 0}};
          MachineSpaceOffset correct =
              MachineSpaceOffset{1, 0, DeviceType::GPU};
          MachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1, 1}};
          MachineSpaceOffset correct =
              MachineSpaceOffset{1, 2, DeviceType::GPU};
          MachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord, ms).value();
          CHECK(correct == result);
        }
      }

      SUBCASE("get_machine_space_offsets") {
        std::unordered_set<MachineSpaceOffset> correct = {
            MachineSpaceOffset{0, 0, DeviceType::GPU},
            MachineSpaceOffset{0, 2, DeviceType::GPU},
            MachineSpaceOffset{1, 0, DeviceType::GPU},
            MachineSpaceOffset{1, 2, DeviceType::GPU}};
        std::unordered_set<MachineSpaceOffset> result =
            get_machine_space_offsets(task, simv, ms);
        CHECK(correct == result);
      }
    }
  }
}
