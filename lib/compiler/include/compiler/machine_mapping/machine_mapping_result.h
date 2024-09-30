#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_RESULT_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_RESULT_H

#include "compiler/machine_mapping/machine_mapping_result.dtg.h"
#include "compiler/machine_mapping/parallel_split_transformation.dtg.h"

namespace FlexFlow {

MachineMappingResult infeasible_machine_mapping_result();
bool is_infeasible(MachineMappingResult const &);
FeasibleMachineMappingResult require_feasible(MachineMappingResult const &);

MachineMappingResult series_combine(float comm_cost,
                    MachineMappingResult const &pre_result,
                    MachineMappingResult const &post_result,
                    std::optional<ParallelSplitTransformation> const &parallel_split_transformation);
MachineMappingResult parallel_combine(MachineMappingResult const &lhs_result,
                                      MachineMappingResult const &rhs_result);

[[nodiscard]] MachineMappingResult minimize_runtime(MachineMappingResult const &m1, MachineMappingResult const &m2);

MachineMappingResult make_singleton_machine_mapping_result(float runtime,
                                                           MachineView const &machine_view);

} // namespace FlexFlow

#endif
