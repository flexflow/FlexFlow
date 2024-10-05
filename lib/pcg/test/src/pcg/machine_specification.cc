#include "pcg/machine_specification.h"
#include "pcg/device_id.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("MachineSpecification") {

    MachineSpecification ms = MachineSpecification{4, 16, 8, 100.0f, 200.0f};

    SUBCASE("get_num_gpus") {
      CHECK(get_num_gpus(ms) == 4 * 8);
    }

    SUBCASE("get_num_cpus") {
      CHECK(get_num_cpus(ms) == 4 * 16);
    }

    SUBCASE("get_num_devices") {
      CHECK(get_num_devices(ms, DeviceType::GPU) == 4 * 8);
      CHECK(get_num_devices(ms, DeviceType::CPU) == 16 * 4);
    }

    SUBCASE("get_device_id") {
      SUBCASE("valid MachineSpaceCoordinate") {
        MachineSpaceCoordinate coord =
            MachineSpaceCoordinate{2, 12, DeviceType::CPU};
        device_id_t correct =
            device_id_from_index(2 * 16 + 12, DeviceType::CPU);
        device_id_t result = get_device_id(ms, coord);
        CHECK(correct == result);
      }
      SUBCASE("invalid MachineSpaceCoordinate") {
        MachineSpaceCoordinate coord =
            MachineSpaceCoordinate{2, 18, DeviceType::CPU};
        CHECK_THROWS(get_device_id(ms, coord));
      }
    }
  }
}
