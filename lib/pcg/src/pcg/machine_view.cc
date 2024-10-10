#include "pcg/machine_view.h"
#include "pcg/machine_specification.h"
#include "pcg/operator_task_space.h"
#include "utils/containers/contains.h"
#include "utils/containers/count.h"
#include "utils/containers/filter.h"
#include "utils/containers/scanl.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"

namespace FlexFlow {

size_t num_dims(MachineView const &mv) {
  return get_strides(mv).size();
}

DeviceType get_device_type(MachineView const &mv) {
  return mv.start.device_type;
}

std::vector<stride_t> get_strides(MachineView const &mv) {
  return transform(mv.dimensions,
                   [](MachineViewDimension const &dim) { return dim.stride; });
}

std::vector<MachineSpecificationDimension>
    get_dimensions(MachineView const &mv) {
  return transform(mv.dimensions, [](MachineViewDimension const &dim) {
    return dim.projection;
  });
}

MachineView machine_view_from_strides_and_machine_spec_dimensions(
    MachineSpaceCoordinate const &start,
    std::vector<stride_t> const &strides,
    std::vector<MachineSpecificationDimension> const &dims) {
  std::vector<MachineViewDimension> dimensions =
      transform(zip(strides, dims), [&](auto const &p) {
        return MachineViewDimension{p.first, p.second};
      });
  return MachineView{start, dimensions};
}

std::optional<MachineSpaceCoordinate> get_machine_space_coordinate(
    OperatorTaskSpace const &task,
    MachineView const &machine_view,
    TaskSpaceCoordinate const &coord,
    MachineSpecification const &machine_specification) {

  auto get_dimension_indices_for_dimension =
      [&](MachineSpecificationDimension dimension) {
        std::vector<MachineSpecificationDimension> mv_dimensions =
            get_dimensions(machine_view);
        return filter(count(mv_dimensions.size()), [&](size_t idx) {
          return mv_dimensions.at(idx) == dimension;
        });
      };

  auto compute_index = [&](int start_idx,
                           std::vector<size_t> const &dimension_indices) {
    std::vector<stride_t> mv_strides = get_strides(machine_view);

    std::vector<int> sizes = transform(dimension_indices, [&](size_t i) {
      return task.degrees.at(i) * mv_strides.at(i).unwrapped;
    });
    std::vector<int> coord_points = transform(
        dimension_indices, [&](size_t i) { return coord.raw_coord.at(i); });
    std::vector<int> strides = transform(dimension_indices, [&](size_t i) {
      return mv_strides.at(i).unwrapped;
    });

    std::vector<int> coeffs = scanl(sizes, 1, std::multiplies<int>());

    int index = start_idx;
    for (auto [coeff, coord_point, stride] :
         zip(coeffs, coord_points, strides)) {
      index += coeff * coord_point * stride;
    }
    return index;
  };

  std::vector<size_t> inter_dimension_indices =
      get_dimension_indices_for_dimension(
          MachineSpecificationDimension::INTER_NODE);
  std::vector<size_t> intra_dimension_indices =
      get_dimension_indices_for_dimension(
          MachineSpecificationDimension::INTRA_NODE);

  int node_idx =
      compute_index(machine_view.start.node_idx, inter_dimension_indices);
  int device_idx =
      compute_index(machine_view.start.device_idx, intra_dimension_indices);
  MachineSpaceCoordinate ms_coord = MachineSpaceCoordinate{
      node_idx, device_idx, get_device_type(machine_view)};

  if (!is_valid_machine_space_coordinate(machine_specification, ms_coord)) {
    return std::nullopt;
  }
  return ms_coord;
}

std::unordered_set<MachineSpaceCoordinate> get_machine_space_coordinates(
    OperatorTaskSpace const &task,
    MachineView const &machine_view,
    MachineSpecification const &machine_specification) {
  return transform(
      get_task_space_coordinates(task), [&](TaskSpaceCoordinate const &coord) {
        return get_machine_space_coordinate(
                   task, machine_view, coord, machine_specification)
            .value();
      });
}

} // namespace FlexFlow
