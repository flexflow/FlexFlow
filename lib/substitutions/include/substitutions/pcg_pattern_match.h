#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_PCG_PATTERN_MATCH_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_PCG_PATTERN_MATCH_H

#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "substitutions/pcg_pattern_match.dtg.h"
#include "substitutions/pcg_pattern.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "substitutions/unlabelled/pattern_node_output.dtg.h"

namespace FlexFlow {

bidict<PatternNodeOutput, parallel_tensor_guid_t> get_output_mapping_for_pcg_pattern_match(PCGPatternMatch const &match, 
                                                                                           PCGPattern const &pattern,
                                                                                           SubParallelComputationGraph const &spcg);

} // namespace FlexFlow

#endif
