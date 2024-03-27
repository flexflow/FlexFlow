#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H

#include "layer.h"
#include "operator_guid_t.h"
#include "tensor.h"
#include "tensor_guid_t.h"
#include "utils/graph.h"
#include "utils/strong_typedef.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct ComputationGraph
    : public strong_typedef<ComputationGraph,
                            OutputLabelledMultiDiGraph<Layer, Tensor>> {
  using strong_typedef::strong_typedef;

  // std::vector<operator_guid_t> get_topological() {
  //   return get_topological_ordering(this->value()); // transform
  // }

  // operator_guid_t add_node(Layer const & layer) {
  //   Node added_node = this->value().add_node(layer);
  //   return operator_guid_t{added_node};
  // }
  // void add_edge(Tensor const &);
  // void add_edge_with_src(Tensor const &, size_t src);
  // MultiDiEdge get_edge(Tensor const &);
  tensor_guid_t get_tensor_guid_t(Tensor edge) {
    NOT_IMPLEMENTED();
  }
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(ComputationGraph);

} // namespace FlexFlow

#endif
