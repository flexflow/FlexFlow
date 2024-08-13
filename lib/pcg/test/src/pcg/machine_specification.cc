#include "pcg/machine_specification.h"
#include "test/utils/doctest.h"

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
  }
}
