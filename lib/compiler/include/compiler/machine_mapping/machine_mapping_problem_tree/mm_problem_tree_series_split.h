#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MM_PROBLEM_TREE_SERIES_SPLIT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MM_PROBLEM_TREE_SERIES_SPLIT_H

#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_series_split.dtg.h"

namespace FlexFlow {

MachineMappingProblemTree get_pre_child(MMProblemTreeSeriesSplit const &);
MachineMappingProblemTree get_post_child(MMProblemTreeSeriesSplit const &);
AbstractedTensorSetMovement const &
    get_abstracted_tensor_movement(MMProblemTreeSeriesSplit const &);

} // namespace FlexFlow

#endif
