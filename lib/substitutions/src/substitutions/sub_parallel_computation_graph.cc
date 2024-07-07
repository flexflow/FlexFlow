#include "substitutions/sub_parallel_computation_graph.h"

namespace FlexFlow {

ParallelLayerAttrs
    get_parallel_layer_attrs(SubParallelComputationGraph const &spcg,
                             Node const &n) {
  return spcg.raw_graph.at(n);
}

PCGOperatorAttrs get_operator_attrs(SubParallelComputationGraph const &spcg,
                                    Node const &n) {
  return get_parallel_layer_attrs(spcg, n).op_attrs;
}

ParallelTensorAttrs
    get_parallel_tensor_attrs(SubParallelComputationGraph const &spcg,
                              OpenDataflowValue const &v) {
  return spcg.raw_graph.at(v);
}

} // namespace FlexFlow
