#include "pcg/start_invariant_machine_view.h"
#include "pcg/strided_rectangle.h"

namespace FlexFlow {

MachineView machine_view_from_start_invariant(
    StartInvariantMachineView const &start_invariant_mv,
    DeviceCoordinates const &start) {
  return MachineView{
      start, start_invariant_mv.rect, start_invariant_mv.device_type};
}
StartInvariantMachineView
    start_invariant_from_machine_view(MachineView const &mv) {
  return StartInvariantMachineView{mv.rect, mv.device_type};
}

StartInvariantMachineView make_1d_start_invariant_machine_view(
    num_points_t num_points, stride_t stride, DeviceType device_type) {
  return StartInvariantMachineView{
      StridedRectangle{{StridedRectangleSide{num_points, stride}}},
      device_type};
}
} // namespace FlexFlow
