#include "substitutions/pcg_pattern.h"
#include "substitutions/operator_pattern/satisfies_pattern.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/tensor_pattern/satisfies_pattern.h"
#include "substitutions/unlabelled/pattern_value.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms.h"

namespace FlexFlow {

static MatchAdditionalCriterion
    pcg_pattern_criteria(PCGPattern const &pattern,
                         SubParallelComputationGraph const &pcg) {
  return MatchAdditionalCriterion{
      [&](PatternNode const &patternNode, Node const &pcgNode) {
        return operator_satisfies_pattern(
            get_operator_attrs(pcg, parallel_layer_guid_t{pcgNode}),
            get_operator_pattern(pattern, patternNode));
      },
      [&](PatternValue const &patternValue, OpenDataflowValue const &pcgValue) {
        return parallel_tensor_satisfies_pattern(
            get_parallel_tensor_attrs(pcg, open_parallel_tensor_guid_t{pcgValue}),
            get_tensor_pattern(pattern, patternValue));
      }};
}

std::vector<PCGPatternMatch>
    find_pattern_matches(PCGPattern const &pattern,
                         SubParallelComputationGraph const &pcg) {
  std::vector<UnlabelledDataflowGraphPatternMatch> unlabelled_matches = find_pattern_matches(get_unlabelled_pattern(pattern),
                                                                                             pcg.raw_graph,
                                                                                             pcg_pattern_criteria(pattern, pcg));
  auto pcg_match_from_unlabelled_match = [](UnlabelledDataflowGraphPatternMatch const &m) {
    return PCGPatternMatch{
      map_values(m.node_assignment,
                 [](Node const &n) { return parallel_layer_guid_t{n}; }),
      map_values(m.input_assignment,
                 [](OpenDataflowValue const &i) { return open_parallel_tensor_guid_t{i}; }),
    };
  };

  return transform(unlabelled_matches, pcg_match_from_unlabelled_match);
}

UnlabelledGraphPattern get_unlabelled_pattern(PCGPattern const &p) {
  return UnlabelledGraphPattern{p.raw_graph};
}

TensorAttributePattern get_tensor_pattern(PCGPattern const &p,
                                          PatternValue const &v) {
  return p.raw_graph.at(raw_open_dataflow_value_from_pattern_value(v));
}

OperatorAttributePattern get_operator_pattern(PCGPattern const &p,
                                              PatternNode const &n) {
  return p.raw_graph.at(n.raw_node);
}

std::vector<PatternNodeOutput> get_pattern_node_outputs(PCGPattern const &pattern,
                                                PatternNode const &node) {
  std::vector<DataflowOutput> raw_outputs = get_outputs(pattern.raw_graph, node.raw_node);

  return transform(raw_outputs, [](DataflowOutput const &o) { return PatternNodeOutput{o}; });
}

bool assignment_satisfies(
    SubParallelComputationGraph const &pcg,
    PCGPattern const &pattern,
    UnlabelledDataflowGraphPatternMatch const &patternMatch) {
  return unlabelled_pattern_does_match(get_unlabelled_pattern(pattern),
                                       pcg.raw_graph,
                                       patternMatch,
                                       pcg_pattern_criteria(pattern, pcg));
}

} // namespace FlexFlow
