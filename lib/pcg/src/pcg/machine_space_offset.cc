#include "pcg/machine_space_offset.h"
#include "utils/exception.h"

namespace FlexFlow {
MachineSpaceOffset get_machine_space_offset_from_coordinate(
    MachineSpaceCoordinate const &start, MachineSpaceCoordinate const &coord) {
  if ((coord.device_idx < start.device_idx) ||
      (coord.node_idx < start.node_idx)) {
    throw mk_runtime_error(fmt::format(
        "One of the coordinates of start {} is greater than one of the "
        "coordinates of coord {}, are you sure you didn't swap them?",
        start,
        coord));
  }
  if (start.device_type != coord.device_type) {
    throw mk_runtime_error(
        fmt::format("{} has different DeviceType from {}", start, coord));
  }

  return MachineSpaceOffset{coord.node_idx - start.node_idx,
                            coord.device_idx - start.device_idx,
                            coord.device_type};
}

} // namespace FlexFlow
