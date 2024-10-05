#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_GET_MACHINE_MAPPING_PROBLEM_TREE_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_GET_MACHINE_MAPPING_PROBLEM_TREE_H

#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"

namespace FlexFlow {

MachineMappingProblemTree
    get_machine_mapping_problem_tree(ParallelComputationGraph const &pcg,
                                     PCGBinarySPDecomposition const &sp);

} // namespace FlexFlow

#endif
