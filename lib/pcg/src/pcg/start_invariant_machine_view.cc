#include "pcg/start_invariant_machine_view.h"
#include "pcg/strided_rectangle.h"

namespace FlexFlow {

MachineView
    to_start_dependent(StartInvariantMachineView const &start_invariant_mv,
                       device_id_t const &start_id) {
  return MachineView{start_id, start_invariant_mv.rect};
}
StartInvariantMachineView to_start_invariant(MachineView const &mv) {
  return StartInvariantMachineView{mv.rect};
}

StartInvariantMachineView
    make_1d_start_invariant_machine_view(num_points_t num_points,
                                         stride_t stride) {
  return StartInvariantMachineView{
      StridedRectangle{{StridedRectangleSide{num_points, stride}}}};
}
} // namespace FlexFlow
