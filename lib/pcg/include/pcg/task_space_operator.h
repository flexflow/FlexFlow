#ifndef _FLEXFLOW_PCG_INCLUDE_TASK_SPACE_OPERATOR_H
#define _FLEXFLOW_PCG_INCLUDE_TASK_SPACE_OPERATOR_H

#include "pcg/task_space_coordinate.dtg.h"
#include "pcg/task_space_operator.dtg.h"
#include <cstddef>
#include <unordered_set>

namespace FlexFlow {

std::unordered_set<TaskSpaceCoordinate>
    get_fragment_coordinates(TaskSpaceOperator const &task);

TaskSpaceCoordinate
    get_maximum_fragment_coordinate(TaskSpaceOperator const &task);

size_t num_dims(TaskSpaceOperator const &task);
size_t num_fragments(TaskSpaceOperator const &task);

} // namespace FlexFlow

#endif
