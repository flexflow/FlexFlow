#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H

#include "device_id.h"
#include "device_type.h"
#include "strided_rectangle.h"
#include "utils/graph.h"
#include "utils/visitable.h"
#include <cstddef>
#include <ostream>
#include <vector>

namespace FlexFlow {

struct MachineView : public use_visitable_cmp<MachineView> {
  MachineView() = delete;
  MachineView(device_id_t const &, StridedRectangle const &);

  std::vector<int> device_ids() const;

  device_id_t at(FFOrdered<num_points_t> const &coord) const;
  StridedRectangleSide at(size_t) const;

public:
  device_id_t start;
  StridedRectangle rect;
};

std::size_t num_dims(MachineView const &);
std::size_t num_devices(MachineView const &);
DeviceType get_device_type(MachineView const &);

MachineView make_1d_machine_view(gpu_id_t start, gpu_id_t stop, int stride = 1);
MachineView make_1d_machine_view(cpu_id_t start, cpu_id_t stop, int stride = 1);
MachineView make_1d_machine_view(device_id_t start,
                                 num_points_t num_points,
                                 int stride = 1);
MachineView make_1d_machine_view(device_id_t start,
                                 side_size_t interval_size,
                                 int stride = 1);

MachineView make_1d_machine_view(device_id_t start, size_t interval_size);

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::MachineView, start, rect);
MAKE_VISIT_HASHABLE(::FlexFlow::MachineView);

#endif
