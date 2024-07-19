#include "pcg/machine_view.h"
#include "pcg/device_id.h"
#include "pcg/strided_rectangle.dtg.h"
#include "pcg/strided_rectangle.h"
#include "pcg/strided_rectangle_side.h"

namespace FlexFlow {

std::vector<device_id_t> device_ids(MachineView const &) {
  NOT_IMPLEMENTED();
}

std::size_t num_dims(MachineView const &mv) {
  return get_num_dims(mv.rect);
}

size_t num_devices(MachineView const &mv) {
  return get_num_points(mv.rect).unwrapped;
}

DeviceType get_device_type(MachineView const &mv) {
  return get_device_type(mv.start);
}

static StridedRectangle make_1d_rect(int start, int stop, int stride) {
  assert(stop > start);
  assert(stride > 0);
  StridedRectangleSide side =
      strided_side_from_size_and_stride(side_size_t{stop - start}, stride);
  StridedRectangle rect =
      StridedRectangle{std::vector<StridedRectangleSide>{side}};
  return rect;
}

MachineView make_1d_machine_view(gpu_id_t start, gpu_id_t stop, int stride) {
  StridedRectangle rect = make_1d_rect(start.gpu_index, stop.gpu_index, stride);
  return MachineView{device_id_t{start}, rect};
}

MachineView make_1d_machine_view(cpu_id_t start, cpu_id_t stop, int stride) {
  StridedRectangle rect = make_1d_rect(start.cpu_index, stop.cpu_index, stride);
  return MachineView{device_id_t{start}, rect};
}

MachineView
    make_1d_machine_view(device_id_t start, device_id_t stop, int stride) {
  assert(get_device_type(start) == get_device_type(stop));
  if (get_device_type(start) == DeviceType::CPU) {
    return make_1d_machine_view(unwrap_cpu(start), unwrap_cpu(stop), stride);
  }
  assert(get_device_type(start) == DeviceType::GPU);
  return make_1d_machine_view(unwrap_gpu(start), unwrap_gpu(stop), stride);
}

static StridedRectangle
    make_1d_rect(int start, num_points_t num_points, int stride) {
  return make_1d_rect(start, start + num_points.unwrapped * stride, stride);
}

MachineView
    make_1d_machine_view(cpu_id_t start, num_points_t num_points, int stride) {
  StridedRectangle rect = make_1d_rect(start.cpu_index, num_points, stride);
  return MachineView{device_id_t{start}, rect};
}

MachineView
    make_1d_machine_view(gpu_id_t start, num_points_t num_points, int stride) {
  StridedRectangle rect = make_1d_rect(start.gpu_index, num_points, stride);
  return MachineView{device_id_t{start}, rect};
}

MachineView make_1d_machine_view(device_id_t start,
                                 num_points_t num_points,
                                 int stride) {
  if (get_device_type(start) == DeviceType::CPU) {
    return make_1d_machine_view(unwrap_cpu(start), num_points, stride);
  } else {
    assert(get_device_type(start) == DeviceType::GPU);
    return make_1d_machine_view(unwrap_gpu(start), num_points, stride);
  }
}

static StridedRectangle
    make_1d_rect(int start, side_size_t interval_size, int stride) {
  return make_1d_rect(start, start + interval_size.unwrapped, stride);
}

MachineView make_1d_machine_view(cpu_id_t start,
                                 side_size_t interval_size,
                                 int stride) {
  StridedRectangle rect = make_1d_rect(start.cpu_index, interval_size, stride);
  return MachineView{device_id_t{start}, rect};
}

MachineView make_1d_machine_view(gpu_id_t start,
                                 side_size_t interval_size,
                                 int stride) {
  StridedRectangle rect = make_1d_rect(start.gpu_index, interval_size, stride);
  return MachineView{device_id_t{start}, rect};
}
MachineView make_1d_machine_view(device_id_t start,
                                 side_size_t interval_size,
                                 int stride) {

  if (get_device_type(start) == DeviceType::CPU) {
    return make_1d_machine_view(unwrap_cpu(start), interval_size, stride);
  } else {
    assert(get_device_type(start) == DeviceType::GPU);
    return make_1d_machine_view(unwrap_gpu(start), interval_size, stride);
  }
}
MachineView make_1d_machine_view(device_id_t start, size_t interval_size) {
  NOT_IMPLEMENTED();
}

/* device_id_t MachineView::at(FFOrdered<num_points_t> const &coord) const { */
/*   size_t offset = this->rect.at(coord); */
/*   return this->start + offset; */
/* } */

} // namespace FlexFlow
