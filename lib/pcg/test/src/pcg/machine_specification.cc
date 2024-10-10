#include "pcg/machine_specification.h"
#include "pcg/device_id.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("MachineSpecification") {

    MachineSpecification ms = MachineSpecification{
        /*num_nodes=*/4,
        /*num_cpus_per_node=*/16,
        /*num_gpus_per_node=*/8,
        /*inter_node_bandwidth=*/0,
        /*intra_node_bandwidth=*/0,
    };

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
        MachineSpaceCoordinate coord = MachineSpaceCoordinate{
            /*node_idx=*/2,
            /*device_idx=*/12,
            DeviceType::CPU,
        };
        device_id_t correct =
            device_id_from_index(2 * 16 + 12, DeviceType::CPU);
        device_id_t result = get_device_id(ms, coord);
        CHECK(correct == result);
      }
      SUBCASE("MachineSpaceCoordinate out of bounds for given machine spec") {
        MachineSpaceCoordinate coord = MachineSpaceCoordinate{
            /*node_idx=*/2,
            /*device_idx=*/18,
            DeviceType::CPU,
        };
        CHECK_THROWS(get_device_id(ms, coord));
      }
    }
  }
}
