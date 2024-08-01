#include "pcg/file_format/v1/graphs.h"
#include "pcg/file_format/v1/graphs/v1_labelled_dataflow_graph.h"
#include "utils/graph/algorithms.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

V1ComputationGraph to_v1(ComputationGraph const &g) {
  return to_v1<LayerAttrs, TensorAttrs>(g.raw_graph);
}

V1ParallelComputationGraph to_v1(ParallelComputationGraph const &g) {
  return to_v1<ParallelLayerAttrs, ParallelTensorAttrs>(g.raw_graph);
}

} // namespace FlexFlow
