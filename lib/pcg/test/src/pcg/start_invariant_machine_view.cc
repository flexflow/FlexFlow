#include "pcg/start_invariant_machine_view.h"
#include "pcg/machine_view.h"
#include "test/utils/doctest.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("StartInvariantMachineView") {

    device_id_t start = device_id_t{gpu_id_t{0}};
    StridedRectangle rect = StridedRectangle{{
        StridedRectangleSide(num_points_t{2}, stride_t{3}),
        StridedRectangleSide(num_points_t{2}, stride_t{2}),
    }};

    SUBCASE("To StartInvariantMachineView") {

      MachineView input = MachineView{start, rect};

      StartInvariantMachineView correct = StartInvariantMachineView{rect};
      StartInvariantMachineView result =
          start_invariant_from_machine_view(input);
      CHECK(correct == result);
    }

    SUBCASE("From StartInvariantMachineView") {

      StartInvariantMachineView input = StartInvariantMachineView{rect};
      MachineView correct = MachineView{start, rect};
      MachineView result = machine_view_from_start_invariant(input, start);
      CHECK(correct == result);
    }

    SUBCASE("To and From") {
      MachineView correct = MachineView{start, rect};
      MachineView result = machine_view_from_start_invariant(
          start_invariant_from_machine_view(correct), start);
      CHECK(correct == result);
    }

    SUBCASE("From and To") {
      StartInvariantMachineView correct = StartInvariantMachineView{rect};
      StartInvariantMachineView result = start_invariant_from_machine_view(
          machine_view_from_start_invariant(correct, start));
      CHECK(correct == result);
    }
  }
}
