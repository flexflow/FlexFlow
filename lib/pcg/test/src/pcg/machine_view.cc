#include "pcg/machine_view.h"
#include "pcg/strided_rectangle.h"
#include "pcg/strided_rectangle_side.h"
#include "test/utils/doctest.h"
#include "utils/containers/transform.h"

std::unordered_multiset<device_id_t>
    make_gpu_device_ids(std::unordered_multiset<int> ids) {
  return transform(ids, [](int id) { return device_id_t(gpu_id_t(id)); });
}

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("MachineView - utility functions") {
    StridedRectangle rect{{StridedRectangleSide(num_points_t(7), stride_t(5)),
                           StridedRectangleSide(num_points_t(10), stride_t(2)),
                           StridedRectangleSide(num_points_t(1), stride_t(4))}};
    gpu_id_t start(1);
    MachineView mv{device_id_t{start}, rect};

    SUBCASE("num_dims") {
      CHECK(num_dims(mv) == 3);
    }
    SUBCASE("num_devices") {
      CHECK(num_devices(mv) == 7 * 10 * 1);
    }
    SUBCASE("get_size") {
      CHECK(get_size(mv) == (7 * 5) * (10 * 2) * (1 * 4));
    }

    SUBCASE("get_side_size_per_dim") {
      std::vector<side_size_t> expected = {
          side_size_t(1 * 4), side_size_t(7 * 5), side_size_t(10 * 2)};
      std::vector<side_size_t> result = get_side_size_per_dim(mv);
      CHECK(expected == result);
    }
    SUBCASE("get_num_devices_per_dim") {
      std::vector<num_points_t> expected = {
          num_points_t(1), num_points_t(7), num_points_t(10)};
      std::vector<num_points_t> result = get_num_devices_per_dim(mv);
      CHECK(expected == result);
    }

    SUBCASE("get_device_type") {
      CHECK(get_device_type(mv) == DeviceType::GPU);
    }
  }

  TEST_CASE("get_device_ids") {

    SUBCASE("2D MachineView") {
      StridedRectangle rect{{
          StridedRectangleSide(num_points_t(2), stride_t(3)),
          StridedRectangleSide(num_points_t(2), stride_t(2)),
      }};
      gpu_id_t start(0);
      MachineView mv{device_id_t{start}, rect};
      SUBCASE("get_device_ids") {
        std::unordered_multiset<device_id_t> expected =
            make_gpu_device_ids({0, 2, 12, 14});
        std::unordered_multiset<device_id_t> result = get_device_ids(mv);
        CHECK(expected == result);
      }
    }
    SUBCASE("3D MachineView") {
      StridedRectangle rect{{
          StridedRectangleSide(num_points_t(1), stride_t(3)),
          StridedRectangleSide(num_points_t(2), stride_t(1)),
          StridedRectangleSide(num_points_t(2), stride_t(2)),
      }};
      gpu_id_t start(1);
      MachineView mv{device_id_t{start}, rect};

      SUBCASE("get_device_ids") {
        std::unordered_multiset<device_id_t> expected =
            make_gpu_device_ids({1, 4, 13, 16});
        std::unordered_multiset<device_id_t> result = get_device_ids(mv);
        CHECK(expected == result);
      }
    }
  }

  TEST_CASE("get_last_device_id") {
    SUBCASE("2D MachineView") {
      StridedRectangle rect{{
          StridedRectangleSide(num_points_t(2), stride_t(3)),
          StridedRectangleSide(num_points_t(2), stride_t(2)),
      }};
      gpu_id_t start(0);
      MachineView mv{device_id_t{start}, rect};

      SUBCASE("get_last_device_id") {
        CHECK(get_last_device_id(mv) == device_id_t(gpu_id_t(14)));
      }
    }

    SUBCASE("3D MachineView") {
      StridedRectangle rect{{
          StridedRectangleSide(num_points_t(1), stride_t(3)),
          StridedRectangleSide(num_points_t(2), stride_t(1)),
          StridedRectangleSide(num_points_t(2), stride_t(2)),
      }};
      gpu_id_t start(1);
      MachineView mv{device_id_t{start}, rect};

      SUBCASE("get_last_device_id") {
        CHECK(get_last_device_id(mv) == device_id_t(gpu_id_t(16)));
      }
    }
  }

  TEST_CASE("make_1d_machine_view - GPU") {

    StridedRectangle rect{{StridedRectangleSide{num_points_t{7}, stride_t{5}}}};
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
  }

  TEST_CASE("make_1d_machine_view - CPU") {
    StridedRectangle rect{
        {StridedRectangleSide{num_points_t{11}, stride_t{4}}}};
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
  }
}
