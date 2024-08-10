#include "pcg/machine_view.h"
#include "pcg/start_invariant_machine_view.h"
#include "test/utils/doctest.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("StartInvariantMachineView") {

      StridedRectangle rect{{
          StridedRectangleSide(num_points_t(2), stride_t{3}),
          StridedRectangleSide(num_points_t(2), stride_t{2}),
      }};
      device_id_t start = device_id_t(gpu_id_t(5));
      MachineView input = MachineView{start, rect};

      MachineView result = to_start_dependent(to_start_invariant(input), start);
      MachineView correct = input;
      CHECK(correct == input);
    }
}
