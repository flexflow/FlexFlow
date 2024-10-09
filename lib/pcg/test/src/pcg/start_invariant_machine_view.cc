#include "pcg/start_invariant_machine_view.h"
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
}
