#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H

#include "layer.h"
#include "operator_guid_t.h"
#include "pcg/layer.h"
#include "pcg/layer_guid_t.h"
#include "pcg/tensor_guid_t.h"
#include "tensor.h"
#include "utils/graph.h"
#include "utils/strong_typedef.h"
#include "utils/type_traits.h"

namespace FlexFlow {

struct ComputationGraph
    : public strong_typedef<ComputationGraph,
                            OutputLabelledMultiDiGraph<Layer, Tensor>> {
  using strong_typedef::strong_typedef;

  /* ComputationGraph(); */

  operator OutputLabelledMultiDiGraphView<Layer, Tensor>() const;

  NodePort port_for_input(size_t);
  NodePort port_for_weight(size_t);
  NodePort port_for_output(size_t);

  size_t input_for_port(NodePort) const;
  size_t weight_for_port(NodePort) const;
  size_t output_for_port(NodePort) const;
  size_t input_for_port(NodePort);
  size_t weight_for_port(NodePort);
  size_t output_for_port(NodePort);

  MultiDiInput get_input_slot(layer_guid_t, size_t);
  MultiDiInput get_weight_slot(layer_guid_t, size_t);
  MultiDiOutput get_output_slot(layer_guid_t, size_t);

  Tensor at(tensor_guid_t) const;
  Layer at(layer_guid_t) const;
};

optional<layer_guid_t> get_layer_with_name(ComputationGraph const &,
                                           std::string const &);
optional<tensor_guid_t> get_tensor_with_name(ComputationGraph const &,
                                             std::string const &);

CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(ComputationGraph);

} // namespace FlexFlow

#endif
