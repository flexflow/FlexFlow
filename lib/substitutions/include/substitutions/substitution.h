#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H

#include "sub_parallel_computation_graph.dtg.h"
#include "substitutions/substitution.dtg.h"
#include "substitutions/unlabelled/multidigraph_pattern_match.dtg.h"

namespace FlexFlow {

bool is_valid_substitution(Substitution const &);

SubParallelComputationGraph
    apply_substitution(SubParallelComputationGraph const &,
                       Substitution const &,
                       MultiDiGraphPatternMatch const &);

} // namespace FlexFlow

#endif
