#include "pcg/operator_task_space.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/maximum.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/fmt/unordered_set.h"

namespace FlexFlow {

std::unordered_set<TaskSpaceCoordinate>
    get_task_space_coordinates(OperatorTaskSpace const &task) {

  std::vector<std::vector<int>> coordinate_ranges = transform(
      task.degrees, [&](int const &num_points) { return range(num_points); });

  std::unordered_set<std::vector<int>> raw_coordinates =
      unordered_set_of(cartesian_product(coordinate_ranges));
  std::unordered_set<TaskSpaceCoordinate> task_space_coordinates =
      transform(raw_coordinates, [](std::vector<int> const &point) {
        return TaskSpaceCoordinate{point};
      });
  return task_space_coordinates;
}

TaskSpaceCoordinate
    get_task_space_maximum_coordinate(OperatorTaskSpace const &task) {
  return maximum(get_task_space_coordinates(task));
}

size_t num_dims(OperatorTaskSpace const &task) {
  return task.degrees.size();
}
size_t num_tasks(OperatorTaskSpace const &task) {
  return product(task.degrees);
}

} // namespace FlexFlow
