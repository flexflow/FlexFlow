#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H

#include "substitutions/pcg_pattern_match.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "substitutions/substitution.dtg.h"

namespace FlexFlow {

/**
 * @brief Checks that all internal invariants of the given substitution hold
 *
 * @details In order for the result of substitution application to be a valid
 * PCG, a Substitution must maintain invariants on the inputs and outputs of
 * both its left-hand side (Substitution::pcg_pattern) and its right-hand side
 * (Substitution::output_graph_expr). More concretely, every Substitution has
 * fields Substitution::input_edge_match_to_output and
 * Substitution::output_edge_match_to_output which must provide a bijection all
 * of the inputs (outputs respectively) of Substitution::pcg_pattern and
 * Substitution::output_graph_expr. If any of these invariants are violated,
 * this function returns false instead of true.
 */
bool is_valid_substitution(Substitution const &);

/**
 * @brief Applies \p substitution to \p sub_pcg at the location specified by \p
 * match, returning the resulting SubParallelComputationGraph
 *
 * @param sub_pcg
 * @param substitution
 * @param match The location at which to apply substitution. This location in
 * sub_pcg should match substitution's PCGPattern. Likely created by running
 * FlexFlow::find_pattern_matches(PCGPattern const &,
 * SubParallelComputationGraph const &).
 * @return SubParallelComputationGraph A sub-PCG similar to sub_pcg, but with
 * the subgraph specified by match replaced with the result of the output
 * expression of substitution
 */
SubParallelComputationGraph
    apply_substitution(SubParallelComputationGraph const &sub_pcg,
                       Substitution const &substitution,
                       PCGPatternMatch const &match);

} // namespace FlexFlow

#endif
