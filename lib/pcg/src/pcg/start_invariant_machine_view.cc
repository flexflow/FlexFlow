#include "pcg/start_invariant_machine_view.h"
#include "pcg/strided_rectangle.h"

namespace FlexFlow {

MachineView machine_view_from_start_invariant(
    StartInvariantMachineView const &start_invariant_mv,
    device_id_t const &start_id) {
  return MachineView{start_id, start_invariant_mv.rect};
}
StartInvariantMachineView
    start_invariant_from_machine_view(MachineView const &mv) {
  return StartInvariantMachineView{mv.rect};
}

StartInvariantMachineView
    make_1d_start_invariant_machine_view(num_points_t num_points,
                                         stride_t stride) {
  return StartInvariantMachineView{
      StridedRectangle{{StridedRectangleSide{num_points, stride}}}};
}
} // namespace FlexFlow
