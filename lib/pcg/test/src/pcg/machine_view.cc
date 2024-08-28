#include "pcg/machine_view.h"
#include "pcg/strided_rectangle.h"
#include "pcg/strided_rectangle_side.h"
#include "test/utils/doctest.h"
#include "utils/containers/transform.h"

std::unordered_set<device_id_t>
    make_gpu_device_ids(std::unordered_set<int> ids) {
  return transform(ids, [](int id) { return device_id_t(gpu_id_t(id)); });
}

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("MachineView - utility functions") {
    MachineView mv = MachineView{
        device_id_t{gpu_id_t{1}},
        StridedRectangle{{StridedRectangleSide(num_points_t{7}, stride_t{5}),
                          StridedRectangleSide(num_points_t{10}, stride_t{2}),
                          StridedRectangleSide(num_points_t{1}, stride_t{4})}}};

    SUBCASE("num_dims") {
      CHECK(num_dims(mv) == 3);
    }
    SUBCASE("num_devices") {
      CHECK(num_devices(mv) == 7 * 10 * 1);
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
      // 2D MachineView describes a 4 x 6 area.
      // The devices are at coordinates (0,0), (0, 3), (2, 0), (2, 3)
      // Thus we have as device ids:
      //  0  = 0*1 + 0*4
      //  12 = 0*1 + 3*4
      //  2  = 2*1 + 0*4
      //  14 = 2*1 + 3*4
      // The coefficients are obtained by doing 
      //`scanl(area_coefficients, 1,product) = {1,4}` 
      // and ignoring the last term.

      MachineView mv =
          MachineView{device_id_t{gpu_id_t{0}},
                      StridedRectangle{{
                          StridedRectangleSide(num_points_t(2), stride_t{3}),
                          StridedRectangleSide(num_points_t(2), stride_t{2}),
                      }}};
      SUBCASE("get_device_ids") {
        std::unordered_set<device_id_t> expected =
            make_gpu_device_ids({0, 2, 12, 14});
        std::unordered_set<device_id_t> result = get_device_ids(mv);
        CHECK(expected == result);
      }
    }
    SUBCASE("3D MachineView") {
      // 3D MachineView describes a 3 x 2 x 4 area, and 1*2*2=4 devices.
      // (Pre offset) the devices are at coordinates (0, 0, 0), (0, 0, 2), (0,
      // 1, 0), (0, 1, 2) Thus (pre offset) we have as device ids:
      //  0  = 0*1 + 0*3 + 0*(2*3)
      //  12  = 0*1 + 0*3 + 2*(2*3)
      //  3  = 0*1 + 1*3 + 0*(2*3)
      //  15  = 0*1 + 1*3 + 1*(2*3)
      // Where the coefficients are obtained by doing `scanl(area_coefficients,
      // 1, product) = {1,3,6}` and ignoring the last term. We do, however, have
      // 1 as a starting device, meaning all device-id are offset by 1. We thus
      // have 1, 13, 4, 16 as device-ids
      MachineView mv =
          MachineView{device_id_t{gpu_id_t{1}},
                      StridedRectangle{{
                          StridedRectangleSide(num_points_t(1), stride_t{3}),
                          StridedRectangleSide(num_points_t(2), stride_t{1}),
                          StridedRectangleSide(num_points_t(2), stride_t{2}),
                      }}};

      SUBCASE("get_device_ids") {
        std::unordered_set<device_id_t> expected =
            make_gpu_device_ids({1, 4, 13, 16});
        std::unordered_set<device_id_t> result = get_device_ids(mv);
        CHECK(expected == result);
      }
    }
  }

  TEST_CASE("get_maximum_device_id") {
    SUBCASE("2D MachineView") {

      MachineView mv =
          MachineView{device_id_t{gpu_id_t{0}},
                      StridedRectangle{{
                          StridedRectangleSide(num_points_t(2), stride_t{3}),
                          StridedRectangleSide(num_points_t(2), stride_t{2}),
                      }}};

      SUBCASE("get_maximum_device_id") {
        CHECK(get_maximum_device_id(mv) == device_id_t(gpu_id_t(14)));
      }
    }

    SUBCASE("3D MachineView") {
      StridedRectangle rect{{
          StridedRectangleSide(num_points_t(1), stride_t{3}),
          StridedRectangleSide(num_points_t(2), stride_t{1}),
          StridedRectangleSide(num_points_t(2), stride_t{2}),
      }};
      MachineView mv{device_id_t{gpu_id_t{1}},
                     StridedRectangle{{
                         StridedRectangleSide(num_points_t(1), stride_t{3}),
                         StridedRectangleSide(num_points_t(2), stride_t{1}),
                         StridedRectangleSide(num_points_t(2), stride_t{2}),
                     }}};

      SUBCASE("get_maximum_device_id") {
        CHECK(get_maximum_device_id(mv) == device_id_t(gpu_id_t(16)));
      }
    }
  }

  TEST_CASE("make_1d_machine_view - GPU") {

    device_id_t start_gpu = device_id_t{gpu_id_t{1}};
    MachineView gpu_mv = MachineView{
        start_gpu,
        StridedRectangle{{StridedRectangleSide{num_points_t{7}, stride_t{5}}}}};

    SUBCASE("make_1d_machine_view(gpu_id_t start, gpu_id_t stop, stride_t "
            "stride)") {
      MachineView result = make_1d_machine_view(
          start_gpu, device_id_t{gpu_id_t(1 + 7 * 5)}, stride_t{5});
      MachineView correct = gpu_mv;
      CHECK(result == correct);
    }

    SUBCASE("make_1d_machine_view(gpu_id_t start, num_points_t num_points, "
            "stride_t stride)") {
      MachineView result =
          make_1d_machine_view(start_gpu, num_points_t{7}, stride_t{5});
      MachineView correct = gpu_mv;
      CHECK(result == correct);
    }
  }

  TEST_CASE("make_1d_machine_view - CPU") {
    device_id_t start_cpu = device_id_t{cpu_id_t{2}};
    MachineView cpu_mv =
        MachineView{start_cpu,
                    StridedRectangle{
                        {StridedRectangleSide{num_points_t{11}, stride_t{4}}}}};

    SUBCASE("make_1d_machine_view(cpu_id_t start, cpu_id_t stop, stride_t "
            "stride)") {
      MachineView result = make_1d_machine_view(
          start_cpu, device_id_t{cpu_id_t(2 + 11 * 4)}, stride_t{4});
      MachineView correct = cpu_mv;
      CHECK(result == correct);
    }
    SUBCASE("make_1d_machine_view(cpu_id_t start, num_points_t num_points, "
            "stride_t stride)") {
      MachineView result =
          make_1d_machine_view(start_cpu, num_points_t{11}, stride_t{4});
      MachineView correct = cpu_mv;
      CHECK(result == correct);
    }
  }
}
