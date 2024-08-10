#include "pcg/machine_view.h"
#include "pcg/device_coordinates.dtg.h"
#include "pcg/device_id.h"
#include "pcg/strided_rectangle.h"
#include "pcg/strided_rectangle_side.h"
#include "utils/containers.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/reversed.h"
#include "utils/containers/scanl.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/zip.h"
#include "utils/hash/vector.h"

namespace FlexFlow {

static device_id_t get_device_id(MachineView const &mv,
                                 DeviceCoordinates const &point) {
  assert(point.coords.size() == get_num_dims(mv.rect));
  std::vector<int> coefficients =
      scanl(mv.rect.get_sides(),
            1,
            [](size_t const &result, StridedRectangleSide const &side) {
              return result * get_side_size(side).unwrapped;
            });
  size_t raw_id =
      sum(transform(zip(coefficients, as_vector(point.coords)),
                    [](auto const pair) { return pair.first * pair.second; })) +
      get_raw_id(mv.start);

  return ((get_device_type(mv) == DeviceType::CPU)
              ? device_id_t(cpu_id_t(raw_id))
              : device_id_t(gpu_id_t(raw_id)));
}

std::unordered_set<device_id_t> get_device_ids(MachineView const &mv) {
  std::vector<std::vector<int>> ranges =
      transform(mv.rect.get_sides(), [](StridedRectangleSide const &side) {
        return range(0, get_side_size(side).unwrapped, side.stride.unwrapped);
      });
  std::unordered_set<DeviceCoordinates> devices_as_points = unordered_set_of(
      transform(cartesian_product(ranges),
                [](auto const &point) { return DeviceCoordinates(point); }));
  std::unordered_set<device_id_t> ids =
      transform(devices_as_points, [&](DeviceCoordinates const &dc) {
        return get_device_id(mv, dc);
      });
  return ids;
}

device_id_t get_last_device_id(MachineView const &mv) {
  // DeviceCoordinates last_device = DeviceCoordinates(
  //     transform(mv.rect.get_sides(), [](StridedRectangleSide const &s) {
  //       return s.stride.unwrapped;
  //     }));
  return maximum(get_device_ids(mv));
}

size_t num_dims(MachineView const &mv) {
  return get_num_dims(mv.rect);
}

std::vector<num_points_t> get_num_devices_per_dim(MachineView const &mv) {
  return transform(mv.rect.get_sides(), [](StridedRectangleSide const &side) {
    return side.num_points;
  });
}

std::vector<side_size_t> get_side_size_per_dim(MachineView const &mv) {
  return transform(mv.rect.get_sides(), get_side_size);
}

size_t num_devices(MachineView const &mv) {
  return get_num_points(mv.rect).unwrapped;
}

size_t get_size(MachineView const &mv) {
  return get_size(mv.rect);
}

DeviceType get_device_type(MachineView const &mv) {
  return get_device_type(mv.start);
}

static StridedRectangle make_1d_rect(int start, int stop, stride_t stride) {
  assert(stop > start);
  assert(stride > stride_t(0));
  StridedRectangleSide side =
      strided_side_from_size_and_stride(side_size_t{stop - start}, stride);
  StridedRectangle rect =
      StridedRectangle{std::vector<StridedRectangleSide>{side}};
  return rect;
}

MachineView
    make_1d_machine_view(gpu_id_t start, gpu_id_t stop, stride_t stride) {
  StridedRectangle rect = make_1d_rect(start.gpu_index, stop.gpu_index, stride);
  return MachineView{device_id_t{start}, rect};
}

MachineView
    make_1d_machine_view(cpu_id_t start, cpu_id_t stop, stride_t stride) {
  StridedRectangle rect = make_1d_rect(start.cpu_index, stop.cpu_index, stride);
  return MachineView{device_id_t{start}, rect};
}

MachineView
    make_1d_machine_view(device_id_t start, device_id_t stop, stride_t stride) {
  assert(get_device_type(start) == get_device_type(stop));
  if (get_device_type(start) == DeviceType::CPU) {
    return make_1d_machine_view(unwrap_cpu(start), unwrap_cpu(stop), stride);
  }
  assert(get_device_type(start) == DeviceType::GPU);
  return make_1d_machine_view(unwrap_gpu(start), unwrap_gpu(stop), stride);
}

static StridedRectangle
    make_1d_rect(int start, num_points_t num_points, stride_t stride) {
  return make_1d_rect(
      start, start + num_points.unwrapped * stride.unwrapped, stride);
}

MachineView make_1d_machine_view(cpu_id_t start,
                                 num_points_t num_points,
                                 stride_t stride) {
  StridedRectangle rect = make_1d_rect(start.cpu_index, num_points, stride);
  return MachineView{device_id_t{start}, rect};
}

MachineView make_1d_machine_view(gpu_id_t start,
                                 num_points_t num_points,
                                 stride_t stride) {
  StridedRectangle rect = make_1d_rect(start.gpu_index, num_points, stride);
  return MachineView{device_id_t{start}, rect};
}

MachineView make_1d_machine_view(device_id_t start,
                                 num_points_t num_points,
                                 stride_t stride) {
  if (get_device_type(start) == DeviceType::CPU) {
    return make_1d_machine_view(unwrap_cpu(start), num_points, stride);
  } else {
    assert(get_device_type(start) == DeviceType::GPU);
    return make_1d_machine_view(unwrap_gpu(start), num_points, stride);
  }
}

static StridedRectangle
    make_1d_rect(int start, side_size_t interval_size, stride_t stride) {
  return make_1d_rect(start, start + interval_size.unwrapped, stride);
}

MachineView make_1d_machine_view(cpu_id_t start,
                                 side_size_t interval_size,
                                 stride_t stride) {
  StridedRectangle rect = make_1d_rect(start.cpu_index, interval_size, stride);
  return MachineView{device_id_t{start}, rect};
}

MachineView make_1d_machine_view(gpu_id_t start,
                                 side_size_t interval_size,
                                 stride_t stride) {
  StridedRectangle rect = make_1d_rect(start.gpu_index, interval_size, stride);
  return MachineView{device_id_t{start}, rect};
}
MachineView make_1d_machine_view(device_id_t start,
                                 side_size_t interval_size,
                                 stride_t stride) {

  if (get_device_type(start) == DeviceType::CPU) {
    return make_1d_machine_view(unwrap_cpu(start), interval_size, stride);
  } else {
    assert(get_device_type(start) == DeviceType::GPU);
    return make_1d_machine_view(unwrap_gpu(start), interval_size, stride);
  }
}

} // namespace FlexFlow
