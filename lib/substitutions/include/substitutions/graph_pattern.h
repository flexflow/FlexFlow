#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTIONS_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTIONS_H

#include "graph_pattern_match.h"
#include "operator_pattern.h"
#include "parallel_tensor_pattern.h"
#include "sub_parallel_computation_graph.h"

namespace FlexFlow {

/**
 * @struct GraphPattern
 * @brief A GraphPattern is defined as an open graph with node label OperatorPattern 
 * and output label ParallelTensorPattern, which is refered to as the pattern graph. 
 * The graph structure of a GraphPattern instance defines the geometrical property 
 * of the input graph, while the node labels and output labels define the attribute 
 * property of that. To be detailed, the OperatorPattern and ParallelTensorPattern 
 * contains a set of constraints and the corresponding graph needs to satisfy these 
 * constraints in order to be considered as match.
 */
struct GraphPattern
    : public strong_typedef<
          GraphPattern,
          OutputLabelledOpenMultiDiGraph<OperatorPattern,
                                         ParallelTensorPattern>> {
  using strong_typedef::strong_typedef;
};

/**
 * @brief Given a pattern, split_pattern is used to split the pattern
 * and recursively match the sub-patterns.
 */
GraphSplit split_pattern(OpenMultiDiGraphView const &pattern);

/**
 * @brief singleton_pattern is defined as a pattern that has only one node.
 * A singleton pattern serves as the base case for recursive pattern matching.
 */
bool is_singleton_pattern(OpenMultiDiGraphView const &);

/**
 * @brief operator_satisfies checks if the operator satisfies the set of constraints.
 * shown in the pattern.
 */
bool operator_satisfies(Operator const &params, OperatorPattern const &pattern);


/**
 * @brief parallel_tensor_satisfies checks if the parallel tensor satisfies the set of 
 * constraints shown in the pattern.
 */
bool parallel_tensor_satisfies(ParallelTensor const &params,
                               ParallelTensorPattern const &pattern);

/**
 * @brief assignment_satifies checks if the provided MultiDiGraphPatternMatch is a valid
 * description of how GraphPattern can be mapped to SubParallelComputationGraph.
 * 
 * It checkes if the node and edge assignments satisfy the constraints of the pattern and whether
 * the graph topology matches.
 */
bool assignment_satisfies(SubParallelComputationGraph const &,
                          GraphPattern const &,
                          MultiDiGraphPatternMatch const &);

} // namespace FlexFlow

#endif
