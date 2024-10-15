#include "pcg/start_invariant_machine_view.h"
#include "pcg/machine_space_offset.h"
#include "pcg/machine_view.h"
#include "pcg/operator_task_space.h"
#include "utils/containers/count.h"
#include "utils/containers/filter.h"
#include "utils/containers/scanl.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"
namespace FlexFlow {

MachineView machine_view_from_start_invariant(
    StartInvariantMachineView const &start_inv_mv,
    MachineSpaceCoordinate const &start) {
  return MachineView{start, start_inv_mv.dimensions};
}

StartInvariantMachineView
    start_invariant_from_machine_view(MachineView const &mv) {
  return StartInvariantMachineView{mv.dimensions, get_device_type(mv)};
}

size_t num_dims(StartInvariantMachineView const &start_inv_mv) {
  return start_inv_mv.dimensions.size();
}

DeviceType get_device_type(StartInvariantMachineView const &start_inv_mv) {
  return start_inv_mv.device_type;
}

std::vector<stride_t>
    get_strides(StartInvariantMachineView const &start_inv_mv) {
  return transform(start_inv_mv.dimensions,
                   [](MachineViewDimension const &dim) { return dim.stride; });
}

std::vector<MachineSpecificationDimension>
    get_dimensions(StartInvariantMachineView const &start_inv_mv) {
  return transform(
      start_inv_mv.dimensions,
      [](MachineViewDimension const &dim) { return dim.projection; });
}

StartInvariantMachineView
    start_invariant_machine_view_from_strides_and_machine_spec_dimensions(
        std::vector<stride_t> const &strides,
        std::vector<MachineSpecificationDimension> const &dims,
        DeviceType device_type) {
  std::vector<MachineViewDimension> dimensions =
      transform(zip(strides, dims), [&](auto const &p) {
        return MachineViewDimension{p.first, p.second};
      });
  return StartInvariantMachineView{dimensions, device_type};
}

std::optional<MachineSpaceOffset> get_machine_space_offset(
    OperatorTaskSpace const &task,
    StartInvariantMachineView const &start_inv_machine_view,
    TaskSpaceCoordinate const &coord,
    MachineSpecification const &machine_specification) {
  MachineSpaceCoordinate dummy_start =
      MachineSpaceCoordinate{0, 0, get_device_type(start_inv_machine_view)};
  MachineView mv =
      machine_view_from_start_invariant(start_inv_machine_view, dummy_start);
  std::optional<MachineSpaceCoordinate> ms_coord =
      get_machine_space_coordinate(task, mv, coord, machine_specification);
  if (ms_coord == std::nullopt) {
    return std::nullopt;
  }
  return get_machine_space_offset_from_coordinate(dummy_start,
                                                  ms_coord.value());
}

std::unordered_set<MachineSpaceOffset> get_machine_space_offsets(
    OperatorTaskSpace const &task,
    StartInvariantMachineView const &start_inv_machine_view,
    MachineSpecification const &machine_specification) {
  return transform(
      get_task_space_coordinates(task), [&](TaskSpaceCoordinate const &coord) {
        return get_machine_space_offset(
                   task, start_inv_machine_view, coord, machine_specification)
            .value();
      });
}

} // namespace FlexFlow
