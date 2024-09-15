#include "pcg/task_space_operator.h"
#include "utils/containers.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"

namespace FlexFlow {

std::unordered_set<TaskSpaceCoordinate>
    get_fragment_coordinates(TaskSpaceOperator const &task) {

  std::vector<std::vector<int>> coordinate_ranges =
      transform(task.degrees, [&](num_points_t const &num_points) {
        return range(num_points.unwrapped);
      });

  std::unordered_set<std::vector<int>> raw_coordinates =
      unordered_set_of(cartesian_product(coordinate_ranges));
  std::unordered_set<TaskSpaceCoordinate> task_space_coordinates =
      transform(raw_coordinates, [](std::vector<int> const &point) {
        return TaskSpaceCoordinate{point};
      });
  return task_space_coordinates;
}

TaskSpaceCoordinate
    get_maximum_fragment_coordinate(TaskSpaceOperator const &task) {
  return maximum(get_fragment_coordinates(task));
}

size_t num_dims(TaskSpaceOperator const &task) {
  return task.degrees.size();
}
size_t num_fragments(TaskSpaceOperator const &task) {
  return product(transform(task.degrees, [&](num_points_t const &num_points) {
    return num_points.unwrapped;
  }));
}

} // namespace FlexFlow
