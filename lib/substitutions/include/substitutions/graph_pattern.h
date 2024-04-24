#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTIONS_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTIONS_H

#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "substitutions/pcg_pattern.dtg.h"
#include "substitutions/unlabelled/pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_node.dtg.h"
#include "substitutions/unlabelled/pattern_matching.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"

namespace FlexFlow {

UnlabelledGraphPattern get_unlabelled_pattern(PCGPattern const &);

TensorAttributePattern get_tensor_pattern(PCGPattern const &, PatternEdge const &);
OperatorAttributePattern get_operator_pattern(PCGPattern const &, PatternNode const &);

bool assignment_satisfies(SubParallelComputationGraph const &,
                          PCGPattern const &,
                          MultiDiGraphPatternMatch const &);

} // namespace FlexFlow

#endif
