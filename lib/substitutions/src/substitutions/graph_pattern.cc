#include "substitutions/graph_pattern.h"
#include "substitutions/operator_pattern/satisfies_pattern.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/tensor_pattern/satisfies_pattern.h"

namespace FlexFlow {

UnlabelledGraphPattern get_unlabelled_pattern(PCGPattern const &p) {
  return UnlabelledGraphPattern{p.raw_graph};
}

TensorAttributePattern get_tensor_pattern(PCGPattern const &p,
                                          PatternEdge const &e) {
  return p.raw_graph.at(e.raw_edge);
}

OperatorAttributePattern get_operator_pattern(PCGPattern const &p,
                                              PatternNode const &n) {
  return p.raw_graph.at(n.raw_node);
}

bool assignment_satisfies(SubParallelComputationGraph const &pcg,
                          PCGPattern const &pattern,
                          MultiDiGraphPatternMatch const &patternMatch) {
  return unlabelled_pattern_does_match(
      get_unlabelled_pattern(pattern),
      pcg.raw_graph,
      patternMatch,
      MatchAdditionalCriterion{
          [&](PatternNode const &patternNode, Node const &pcgNode) {
            return operator_satisfies_pattern(
                get_operator_attrs(pcg, pcgNode),
                get_operator_pattern(pattern, patternNode));
          },
          [&](PatternEdge const &patternEdge, OpenMultiDiEdge const &pcgEdge) {
            return parallel_tensor_satisfies_pattern(
                get_parallel_tensor_attrs(pcg, pcgEdge),
                get_tensor_pattern(pattern, patternEdge));
          }});
}

} // namespace FlexFlow
