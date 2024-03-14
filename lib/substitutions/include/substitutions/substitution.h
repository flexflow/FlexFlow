#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H

#include "graph_pattern.h"
#include "output_graph.h"
#include "sub_parallel_computation_graph.h"

namespace FlexFlow {

/**
 * @struct Substitution
 * @brief A substitution is to replace a subgraph of the PCG by a new one. 
 * We refer to the subgraph to be replaced as the input graph, and the new 
 * subgraph to replace the input graph as the output graph.
 * A Substitution object describes a substitution. It consists of An 
 * input_graph of type GraphPattern that describes which kind of input graphs 
 * the substitution can be applied to; An output_graph of type OutputGraphExpr 
 * that describes how the output graph is computed from the input graph; and
 * An input_mapping and output_maping that describes how the output graph is 
 * connected to the original PCG.
 */
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

/**
 * @brief is_valid_substitution checks if the substitution is valid. 
 * The implementation will enumerate all the possible substitutions and filter 
 * out all the invalid ones.
 */
bool is_valid_substitution(Substitution const &);


SubParallelComputationGraph
    apply_substitution(SubParallelComputationGraph const &,
                       Substitution const &,
                       MultiDiGraphPatternMatch const &);

} // namespace FlexFlow

#endif
