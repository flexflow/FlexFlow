#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUBSTITUTION_INTERNAL_EVALUATE_SUBSTITUTION_OUTPUT_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUBSTITUTION_INTERNAL_EVALUATE_SUBSTITUTION_OUTPUT_H

#include "substitutions/pcg_pattern_match.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "substitutions/substitution.dtg.h"
#include "substitutions/substitution_internal/output_expr_to_result_sub_pcg_mapping.dtg.h"
#include <utility>

namespace FlexFlow {

/**
 * @brief Takes a SubParallelComputationGraph and a PCGPatternMatch where a
 * Substitution applies and evaluates the Substitution's OutputGraphExpr
 * (producing another SubParallelComputationGraph) using the information from
 * the matched nodes.
 *
 * @details Exists only to enable apply_substitution(SubParallelComputationGraph
 * const &, Substitution const &, PCGPatternMatch const &)
 *
 * @note The resulting SubParallelComputationGraph has new node ids, i.e., does
 * not have the same node ids as the OutputGraphExpr
 */
std::pair<SubParallelComputationGraph, OutputExprToResultSubPCGMapping>
    evaluate_substitution_output(SubParallelComputationGraph const &spcg,
                                 Substitution const &sub,
                                 PCGPatternMatch const &match);

} // namespace FlexFlow

#endif
