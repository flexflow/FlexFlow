#include "pcg/machine_view.h"
#include "pcg/strided_rectangle_side.h"
#include "pcg/strided_rectangle.dtg.h"

namespace FlexFlow {

std::vector<device_id_t> device_ids(MachineView const &) {
  NOT_IMPLEMENTED();
}

std::size_t num_dims(MachineView const &) {
  NOT_IMPLEMENTED();
}

std::size_t num_devices(MachineView const &) { 
  NOT_IMPLEMENTED();
}

DeviceType get_device_type(MachineView const &) {
  NOT_IMPLEMENTED();
}

static StridedRectangle make_1d_rect(int start, int stop, int stride) {
  assert(stop > start);
  assert(stride > 0);
  StridedRectangleSide side = strided_side_from_size_and_stride(side_size_t{stop - start}, stride);
  StridedRectangle rect = {{side}};
  return rect;
}

MachineView make_1d_machine_view(gpu_id_t start, gpu_id_t stop, int stride) {
  StridedRectangle rect = make_1d_rect(start.gpu_index, stop.gpu_index, stride);
  return {device_id_t{start}, rect};
}

MachineView make_1d_machine_view(cpu_id_t start, cpu_id_t stop, int stride) {
  StridedRectangle rect = make_1d_rect(start.cpu_index, stop.cpu_index, stride);
  return {device_id_t{start}, rect};
}

MachineView make_1d_machine_view(device_id_t start,
                                 num_points_t num_points,
                                 int stride) {
  NOT_IMPLEMENTED();
}

MachineView make_1d_machine_view(device_id_t start,
                                 side_size_t interval_size,
                                 int stride) {
  NOT_IMPLEMENTED();
}

MachineView make_1d_machine_view(device_id_t start, size_t interval_size) {
  NOT_IMPLEMENTED();
}

/* device_id_t MachineView::at(FFOrdered<num_points_t> const &coord) const { */
/*   size_t offset = this->rect.at(coord); */
/*   return this->start + offset; */
/* } */


} // namespace FlexFlow
