#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H

#include "substitutions/sub_parallel_computation_graph.dtg.h"

namespace FlexFlow {

ParallelLayerAttrs get_parallel_layer_attrs(SubParallelComputationGraph const &,
                                            Node const &);
PCGOperatorAttrs get_operator_attrs(SubParallelComputationGraph const &,
                                    Node const &);
ParallelTensorAttrs
    get_parallel_tensor_attrs(SubParallelComputationGraph const &,
                              OpenMultiDiEdge const &);

} // namespace FlexFlow

#endif
