#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_RESULT_TREE_MACHINE_MAPPING_RESULT_TREE_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_RESULT_TREE_MACHINE_MAPPING_RESULT_TREE_H

#include "compiler/machine_mapping/machine_mapping_result_tree/machine_mapping_result_tree.dtg.h"

namespace FlexFlow {

MachineMappingResultTree make_series_split(float comm_cost,
                                           MachineMappingResultTree const &pre,
                                           MachineMappingResultTree const &post);
MachineMappingResultTree make_parallel_split(MachineMappingResultTree const &lhs,
                                             MachineMappingResultTree const &rhs);
MachineMappingResultTree make_leaf_node(float cost, MachineView const &);

std::optional<MachineMappingResultTree> minimize_cost(std::optional<MachineMappingResultTree> const &, MachineMappingResultTree const &);

} // namespace FlexFlow

#endif
