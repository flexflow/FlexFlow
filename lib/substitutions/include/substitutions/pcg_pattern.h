#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_PCG_PATTERN_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_PCG_PATTERN_H

#include "substitutions/pcg_pattern.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "substitutions/unlabelled/pattern_value.dtg.h"
#include "substitutions/unlabelled/pattern_matching.h"
#include "substitutions/unlabelled/pattern_node.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"

namespace FlexFlow {

std::vector<UnlabelledDataflowGraphPatternMatch> 
  find_pattern_matches(PCGPattern const &pattern,
                       SubParallelComputationGraph const &pcg);

UnlabelledGraphPattern get_unlabelled_pattern(PCGPattern const &);

TensorAttributePattern get_tensor_pattern(PCGPattern const &,
                                          PatternValue const &);
OperatorAttributePattern get_operator_pattern(PCGPattern const &,
                                              PatternNode const &);
std::unordered_set<PatternInput> get_inputs(PCGPattern const &);

bool assignment_satisfies(SubParallelComputationGraph const &,
                          PCGPattern const &,
                          UnlabelledDataflowGraphPatternMatch const &);

} // namespace FlexFlow

#endif
