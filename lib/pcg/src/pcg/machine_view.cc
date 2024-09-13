#include "pcg/machine_view.h"
#include "pcg/device_id.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view_coordinates.dtg.h"
#include "pcg/machine_view_dim_idx_t.dtg.h"
#include "pcg/machine_view_projection.dtg.h"
#include "pcg/strided_rectangle.h"
#include "pcg/strided_rectangle_side.h"
#include "utils/containers.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/keys.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/reversed.h"
#include "utils/containers/scanl.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/zip.h"
#include "utils/hash/vector.h"

namespace FlexFlow {

std::unordered_set<MachineViewCoordinates>
    get_devices_coordinates(MachineView const &mv) {

  std::vector<std::vector<int>> coordinate_ranges =
      transform(mv.rect.get_sides(), [&](StridedRectangleSide const &side) {
        return range(side.num_points.unwrapped);
      });

  std::unordered_set<std::vector<int>> raw_coordinates =
      unordered_set_of(cartesian_product(coordinate_ranges));
  std::unordered_set<MachineViewCoordinates> machine_view_coordinates =
      transform(raw_coordinates, [](std::vector<int> const &point) {
        return MachineViewCoordinates(point);
      });
  return machine_view_coordinates;
}

MachineViewCoordinates get_maximum_device_coordinates(MachineView const &mv) {
  return maximum(get_devices_coordinates(mv));
}

MachineSpecificationCoordinates get_machine_specification_coordinates(
    MachineView const &mv,
    MachineViewCoordinates const &coordinates,
    MachineSpecification const &ms,
    MachineViewProjection const &projection) {

  auto inter_projection = filter_values(
      projection.raw_projection, [](MachineSpecificationDimension const &dim) {
        return dim == MachineSpecificationDimension::INTER;
      });
  auto intra_projection = filter_values(
      projection.raw_projection, [](MachineSpecificationDimension const &dim) {
        return dim == MachineSpecificationDimension::INTRA;
      });

  MachineViewCoordinates transformed_coordinates = MachineViewCoordinates{
      transform(zip(coordinates.raw_coords, mv.rect.get_sides()),
                [&](auto const &pair) {
                  return pair.first * pair.second.stride.unwrapped;
                })};
  transformed_coordinates = MachineViewCoordinates{
      transform(zip(transformed_coordinates.raw_coords, mv.start.raw_coords),
                [&](auto const &pair) { return pair.first + pair.second; })};

  auto get_coordinate = [&](auto const &sub_projection) {
    std::vector<machine_view_dim_idx_t> relevant_dimensions =
        sorted(keys(sub_projection));
    std::vector<side_size_t> relevant_side_sizes =
        transform(relevant_dimensions, [&](auto const &idx) {
          return get_side_size(get_side_at_idx(mv, idx));
        });
    std::vector<int> coefficients =
        scanl(relevant_side_sizes,
              1,
              [](size_t const &result, side_size_t const &side_size) {
                return result * side_size.unwrapped;
              });
    std::vector<int> filtered_coord;
    for (int i = 0; i < transformed_coordinates.raw_coords.size(); ++i) {
      if (contains(relevant_dimensions, machine_view_dim_idx_t{i})) {
        filtered_coord.push_back(transformed_coordinates.raw_coords[i]);
      }
    }
    return sum(
        transform(zip(coefficients, filtered_coord),
                  [](auto const pair) { return pair.first * pair.second; }));
  };
  int inter_coordinate = get_coordinate(inter_projection);
  int intra_coordinate = get_coordinate(intra_projection);
  return MachineSpecificationCoordinates{
      inter_coordinate, intra_coordinate, mv.device_type};
}

device_id_t get_device_id(MachineView const &mv,
                          MachineViewCoordinates const &coordinates,
                          MachineSpecification const &ms,
                          MachineViewProjection const &projection) {
  MachineSpecificationCoordinates coords =
      get_machine_specification_coordinates(mv, coordinates, ms, projection);
  return get_device_id(ms, coords);
}

std::unordered_set<device_id_t>
    get_device_ids(MachineView const &mv,
                   MachineSpecification const &ms,
                   MachineViewProjection const &projection) {
  std::unordered_set<device_id_t> devices_ids;
  for (MachineViewCoordinates const &coordinates :
       get_devices_coordinates(mv)) {
    devices_ids.insert(get_device_id(mv, coordinates, ms, projection));
  }
  return devices_ids;
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

StridedRectangleSide get_side_at_idx(MachineView const &mv,
                                     machine_view_dim_idx_t const &idx) {
  return mv.rect.at(idx.unwrapped);
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

MachineView make_1d_machine_view(int start,
                                 int stop,
                                 stride_t stride,
                                 DeviceType device_type) {
  StridedRectangle rect = make_1d_rect(start, stop, stride);
  MachineViewCoordinates start_coordinate = MachineViewCoordinates{{start}};
  return MachineView{start_coordinate, rect, device_type};
}

static StridedRectangle
    make_1d_rect(int start, num_points_t num_points, stride_t stride) {
  return make_1d_rect(
      start, start + num_points.unwrapped * stride.unwrapped, stride);
}

MachineView make_1d_machine_view(int start,
                                 num_points_t num_points,
                                 stride_t stride,
                                 DeviceType device_type) {
  StridedRectangle rect = make_1d_rect(start, num_points, stride);
  MachineViewCoordinates start_coordinate = MachineViewCoordinates{{start}};
  return MachineView{start_coordinate, rect, device_type};
}

static StridedRectangle
    make_1d_rect(int start, side_size_t interval_size, stride_t stride) {
  return make_1d_rect(start, start + interval_size.unwrapped, stride);
}

MachineView make_1d_machine_view(int start,
                                 side_size_t interval_size,
                                 stride_t stride,
                                 DeviceType device_type) {
  StridedRectangle rect = make_1d_rect(start, interval_size, stride);
  MachineViewCoordinates start_coordinate = MachineViewCoordinates{{start}};
  return MachineView{start_coordinate, rect, device_type};
}

} // namespace FlexFlow
