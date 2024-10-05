#ifndef _FLEXFLOW_PCG_INCLUDE_operator_task_space_H
#define _FLEXFLOW_PCG_INCLUDE_operator_task_space_H

#include "pcg/operator_task_space.dtg.h"
#include "pcg/task_space_coordinate.dtg.h"
#include <cstddef>
#include <unordered_set>

namespace FlexFlow {

std::unordered_set<TaskSpaceCoordinate>
    get_task_space_coordinates(OperatorTaskSpace const &task);

TaskSpaceCoordinate
    get_task_space_maximum_coordinate(OperatorTaskSpace const &task);

size_t num_dims(OperatorTaskSpace const &task);
size_t num_tasks(OperatorTaskSpace const &task);

} // namespace FlexFlow

#endif
