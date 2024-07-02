#include "doctest/doctest.h"
#include "pcg/machine_view.h"
#include "pcg/strided_rectangle.h"
#include "pcg/strided_rectangle_side.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MachineView general util functions") {
    StridedRectangle rect{{StridedRectangleSide{num_points_t{7}, 5},
                           StridedRectangleSide{num_points_t{10}, 2}}};
    gpu_id_t start(1);
    MachineView mv{device_id_t{start}, rect};
    CHECK(num_dims(mv) == 2);
    CHECK(num_devices(mv) == 7 * 10);
    CHECK(get_device_type(mv) == DeviceType::GPU);
  }

  TEST_CASE("MachineView make_1d_machine_view") {
    StridedRectangle rect{{StridedRectangleSide{num_points_t{7}, 5}}};
    device_id_t start_gpu{gpu_id_t{1}};
    device_id_t start_cpu{cpu_id_t{3}};

    MachineView gpu_mv{start_gpu, rect};
    MachineView cpu_mv{start_cpu, rect};

    CHECK(make_1d_machine_view(
              start_gpu, device_id_t{gpu_id_t(1 + 7 * 5)}, 5) == gpu_mv);
    CHECK(make_1d_machine_view(start_gpu, num_points_t{7}, 5) == gpu_mv);
    CHECK(make_1d_machine_view(start_gpu,
                               get_side_size(rect.sides.at(ff_dim_t{0})),
                               5) == gpu_mv);

    CHECK(make_1d_machine_view(
              start_cpu, device_id_t{cpu_id_t(3 + 7 * 5)}, 5) == cpu_mv);
    CHECK(make_1d_machine_view(start_cpu, num_points_t{7}, 5) == cpu_mv);
    CHECK(make_1d_machine_view(start_cpu,
                               get_side_size(rect.sides.at(ff_dim_t{0})),
                               5) == cpu_mv);
  }
}
