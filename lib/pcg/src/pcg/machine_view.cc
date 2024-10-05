#include "pcg/machine_view.h"
#include "pcg/machine_specification.h"
#include "pcg/operator_task_space.h"
#include "utils/containers/contains.h"
#include "utils/containers/scanl.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"

namespace FlexFlow {

static std::vector<int>
    get_projection_indices(MachineView const &mv,
                           MachineSpecificationDimension dimension) {

  std::vector<int> projection_indices;
  for (size_t i = 0; i < mv.projection.size(); ++i) {
    if (mv.projection[i] == dimension) {
      projection_indices.push_back(i);
    }
  }
  return projection_indices;
}

static int compute_index(int start_idx,
                         std::vector<int> const &projection_indices,
                         OperatorTaskSpace const &task,
                         MachineView const &mv,
                         TaskSpaceCoordinate const &coord) {

  std::vector<int> sizes;
  std::vector<int> coord_points;
  std::vector<int> strides;

  for (int i : projection_indices) {
    int dim_size = task.degrees[i] * mv.strides[i].unwrapped;
    sizes.push_back(dim_size);
    coord_points.push_back(coord.raw_coord[i]);
    strides.push_back(mv.strides[i].unwrapped);
  }

  std::vector<int> coeffs = scanl(sizes, 1, std::multiplies<int>());

  int index = start_idx;
  for (auto [coeff, coord_point, stride] : zip(coeffs, coord_points, strides)) {
    index += coeff * coord_point * stride;
  }
  return index;
}

std::optional<MachineSpaceCoordinate>
    get_machine_space_coordinate(OperatorTaskSpace const &task,
                                 MachineView const &mv,
                                 TaskSpaceCoordinate const &coord,
                                 MachineSpecification const &ms) {

  std::vector<int> inter_projection_indices =
      get_projection_indices(mv, MachineSpecificationDimension::INTER_NODE);
  std::vector<int> intra_projection_indices =
      get_projection_indices(mv, MachineSpecificationDimension::INTRA_NODE);

  int node_idx = compute_index(
      mv.start.node_idx, inter_projection_indices, task, mv, coord);
  int device_idx = compute_index(
      mv.start.device_idx, intra_projection_indices, task, mv, coord);
  MachineSpaceCoordinate ms_coord =
      MachineSpaceCoordinate{node_idx, device_idx, get_device_type(mv)};
  if (!is_valid_machine_space_coordinate(ms, ms_coord)) {
    return std::nullopt;
  }
  return ms_coord;
}
std::unordered_set<MachineSpaceCoordinate>
    get_machine_space_coordinates(OperatorTaskSpace const &task,
                                  MachineView const &mv,
                                  MachineSpecification const &ms) {

  return transform(
      get_task_space_coordinates(task), [&](TaskSpaceCoordinate const &c) {
        return get_machine_space_coordinate(task, mv, c, ms).value();
      });
}

size_t num_dims(MachineView const &mv) {
  return mv.strides.size();
}

DeviceType get_device_type(MachineView const &mv) {
  return mv.start.device_type;
}
} // namespace FlexFlow
