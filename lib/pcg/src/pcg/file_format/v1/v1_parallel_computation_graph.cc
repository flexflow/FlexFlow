#include "pcg/file_format/v1/v1_parallel_computation_graph.h"
#include "pcg/file_format/v1/graphs/v1_labelled_dataflow_graph.h"

namespace FlexFlow {

V1ParallelComputationGraph to_v1(ParallelComputationGraph const &g) {
  return V1ParallelComputationGraph{
    to_v1<ParallelLayerAttrs, ParallelTensorAttrs>(g.raw_graph),
  };
}

} // namespace FlexFlow
