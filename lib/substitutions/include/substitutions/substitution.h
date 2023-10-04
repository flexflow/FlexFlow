#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H

#include "graph_pattern.h"
#include "output_graph.h"
#include "sub_parallel_computation_graph.h"

namespace FlexFlow {

struct Substitution {
  using InputPatternInput = InputMultiDiEdge;
  using InputPatternOutput = OutputMultiDiEdge;
  using OutputPatternInput = InputMultiDiEdge;
  using OutputPatternOutput = OutputMultiDiEdge;

  GraphPattern input_graph;
  OutputGraphExpr output_graph_expr;
  bidict<InputPatternInput, OutputPatternInput> input_mapping;
  bidict<InputPatternOutput, OutputPatternOutput> output_mapping;
};

bool is_valid_substitution(Substitution const &);

SubParallelComputationGraph
    apply_substitution(SubParallelComputationGraph const &,
                       Substitution const &,
                       MultiDiGraphPatternMatch const &);

} // namespace FlexFlow

#endif
