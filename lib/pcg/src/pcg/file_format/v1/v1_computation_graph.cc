#include "pcg/file_format/v1/v1_computation_graph.h"
#include "pcg/file_format/v1/graphs/v1_labelled_dataflow_graph.h"

namespace FlexFlow {

V1ComputationGraph to_v1(ComputationGraph const &g) {
  return V1ComputationGraph{
    to_v1<LayerAttrs, TensorAttrs>(g.raw_graph),
  };
}

} // namespace FlexFlow
