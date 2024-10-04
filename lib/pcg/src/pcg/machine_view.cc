#include "pcg/machine_view.h"
#include "pcg/task_space_operator.h"
#include "utils/containers/contains.h"
#include "utils/containers/scanl.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"

namespace FlexFlow {

MachineSpaceCoordinate
    get_machine_space_coordinate(TaskSpaceOperator const &task,
                                 MachineView const &mv,
                                 TaskSpaceCoordinate const &coord,
                                 MachineSpecification const &ms) {

  std::vector<int> inter_projection;
  std::vector<int> intra_projection;
  for (size_t i = 0; i < num_dims(mv); ++i) {
    if (mv.projection[i] == MachineSpecificationDimension::INTER_NODE) {
      inter_projection.push_back(i);
    } else if (mv.projection[i] == MachineSpecificationDimension::INTRA_NODE) {
      intra_projection.push_back(i);
    }
  }

  std::vector<int> inter_sizes;
  std::vector<int> intra_sizes;
  std::vector<int> inter_coord_points;
  std::vector<int> intra_coord_points;
  std::vector<int> inter_strides;
  std::vector<int> intra_strides;

  for (size_t i = 0; i < num_dims(mv); ++i) {
    int dim_size = task.degrees.at(i).unwrapped * mv.strides.at(i).unwrapped;
    if (contains(inter_projection, i)) {
      inter_sizes.push_back(dim_size);
      inter_coord_points.push_back(coord.raw_coord.at(i));
      inter_strides.push_back(mv.strides.at(i).unwrapped);
    }
    if (contains(intra_projection, i)) {
      intra_sizes.push_back(dim_size);
      intra_coord_points.push_back(coord.raw_coord.at(i));
      intra_strides.push_back(mv.strides.at(i).unwrapped);
    }
  }

  std::vector<int> inter_coeffs = scanl(inter_sizes, 1, std::multiplies<int>());
  std::vector<int> intra_coeffs = scanl(intra_sizes, 1, std::multiplies<int>());

  int inter =
      mv.start.inter +
      sum(transform(zip(inter_coeffs, inter_coord_points, inter_strides),
                    [](auto const &tuple) {
                      return std::get<0>(tuple) * std::get<1>(tuple) *
                             std::get<2>(tuple);
                    }));
  int intra =
      mv.start.intra +
      sum(transform(zip(intra_coeffs, intra_coord_points, intra_strides),
                    [](auto const &tuple) {
                      return std::get<0>(tuple) * std::get<1>(tuple) *
                             std::get<2>(tuple);
                    }));

  return MachineSpaceCoordinate{inter, intra, get_device_type(mv)};
}

std::unordered_set<MachineSpaceCoordinate>
    get_machine_space_coordinates(TaskSpaceOperator const &task,
                                  MachineView const &mv,
                                  MachineSpecification const &ms) {

  return transform(get_fragment_coordinates(task),
                   [&](TaskSpaceCoordinate const &c) {
                     return get_machine_space_coordinate(task, mv, c, ms);
                   });
}

size_t num_dims(MachineView const &mv) {
  return mv.strides.size();
}

DeviceType get_device_type(MachineView const &mv) {
  return mv.start.device_type;
}
} // namespace FlexFlow
