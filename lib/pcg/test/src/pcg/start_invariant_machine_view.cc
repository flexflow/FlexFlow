#include "pcg/start_invariant_machine_view.h"
#include "pcg/machine_view.h"
#include "test/utils/doctest.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("StartInvariantMachineView") {

    MachineViewCoordinate start = MachineViewCoordinate{{0}};
    StridedRectangle rect = StridedRectangle{{
        StridedRectangleSide(num_points_t{2}, stride_t{3}),
        StridedRectangleSide(num_points_t{2}, stride_t{2}),
    }};

    DeviceType device_type = DeviceType::GPU;

    SUBCASE("start_invariant_from_machine_view") {

      MachineView input = MachineView{start, rect, device_type};

      StartInvariantMachineView correct =
          StartInvariantMachineView{rect, device_type};
      StartInvariantMachineView result =
          start_invariant_from_machine_view(input);
      CHECK(correct == result);
    }

    SUBCASE("conversion is invertible") {
      SUBCASE("MachineView -> StrideInvariant -> MachineView") {
        MachineView correct = MachineView{start, rect, device_type};
        MachineView result = machine_view_from_start_invariant(
            start_invariant_from_machine_view(correct), start);
        CHECK(correct == result);
      }

      SUBCASE("StrideInvariant -> MachineView -> StrideInvariant") {
        StartInvariantMachineView correct =
            StartInvariantMachineView{rect, device_type};
        StartInvariantMachineView result = start_invariant_from_machine_view(
            machine_view_from_start_invariant(correct, start));
        CHECK(correct == result);
      }
    }
  }
}
