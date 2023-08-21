#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H

#include "graph_pattern.h"
#include "output_graph.h"

namespace FlexFlow {

struct Substitution {
  GraphPattern input_graph;
  OutputGraph output_graph;
  bidict<InputMultiDiEdge, IutputMultiDiEdge> input_mapping;
  bidict<OutputMultiDiEdge, OutputMultiDiEdge> output_mapping;
};

ParallelComputationGraph apply_substitution(ParallelComputationGraph const &,
                                            Substitution const &,
                                            DiGraphPatternMatch const &);

} // namespace FlexFlow

#endif
