#include "test/utils/doctest.h"
#include "pcg/machine_view.h"
#include "pcg/strided_rectangle.h"
#include "pcg/strided_rectangle_side.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MachineView general util functions") {
    StridedRectangle rect{{StridedRectangleSide{num_points_t{7}, 5},
                           StridedRectangleSide{num_points_t{10}, 2}}};
    gpu_id_t start(1);
    MachineView mv{device_id_t{start}, rect};
    SUBCASE("num_dims") {
      CHECK(num_dims(mv) == 2);
    }
    SUBCASE("num_devices") {
      CHECK(num_devices(mv) == 7 * 10);
    }
    SUBCASE("get_device_type") {
      CHECK(get_device_type(mv) == DeviceType::GPU);
    }
  }

  TEST_CASE("MachineView make_1d_machine_view - GPU") {
    StridedRectangle rect{{StridedRectangleSide{num_points_t{7}, 5}}};
    device_id_t start_gpu{gpu_id_t{1}};
    MachineView gpu_mv{start_gpu, rect};

    SUBCASE("make_1d_machine_view(gpu_id_t start, gpu_id_t stop, int stride)") {
      MachineView result =
          make_1d_machine_view(start_gpu, device_id_t{gpu_id_t(1 + 7 * 5)}, 5);
      MachineView correct = gpu_mv;
      CHECK(result == correct);
    }
    SUBCASE("make_1d_machine_view(gpu_id_t start, num_points_t num_points, int "
            "stride)") {
      MachineView result = make_1d_machine_view(start_gpu, num_points_t{7}, 5);
      MachineView correct = gpu_mv;
      CHECK(result == correct);
    }
    SUBCASE("make_1d_machine_view(gpu_id_t start, side_size_t interval_size, "
            "int stride)") {
      MachineView result = make_1d_machine_view(
          start_gpu, get_side_size(rect.sides.at(ff_dim_t{0})), 5);
      MachineView correct = gpu_mv;
      CHECK(result == correct);
    }
  }

  TEST_CASE("MachineView make_1d_machine_view - CPU") {
    StridedRectangle rect{{StridedRectangleSide{num_points_t{11}, 4}}};
    device_id_t start_cpu{cpu_id_t{2}};
    MachineView cpu_mv{start_cpu, rect};

    SUBCASE("make_1d_machine_view(cpu_id_t start, cpu_id_t stop, int stride)") {
      MachineView result =
          make_1d_machine_view(start_cpu, device_id_t{cpu_id_t(2 + 11 * 4)}, 4);
      MachineView correct = cpu_mv;
      CHECK(result == correct);
    }
    SUBCASE("make_1d_machine_view(cpu_id_t start, num_points_t num_points, int "
            "stride)") {
      MachineView result = make_1d_machine_view(start_cpu, num_points_t{11}, 4);
      MachineView correct = cpu_mv;
      CHECK(result == correct);
    }
    SUBCASE("make_1d_machine_view(cpu_id_t start, side_size_t interval_size, "
            "int stride)") {
      MachineView result = make_1d_machine_view(
          start_cpu, get_side_size(rect.sides.at(ff_dim_t{0})), 4);
      MachineView correct = cpu_mv;
      CHECK(result == correct);
    }
  }
}
