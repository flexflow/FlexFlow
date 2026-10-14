#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_labelled_open_kwarg_dataflow_graph_labels.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using GraphInputName = ordered_value_type<2>;
using SlotName = ordered_value_type<3>;
using NewNodeLabel = value_type<4>;
using NewValueLabel = value_type<5>;

struct F {
  NewNodeLabel operator()(Node const &, NodeLabel const &) {
    NOT_IMPLEMENTED();
  }
  NewValueLabel
      operator()(OpenKwargDataflowValue<GraphInputName, SlotName> const &,
                 ValueLabel) {
    NOT_IMPLEMENTED();
  }
};

template LabelledOpenKwargDataflowGraphView<NewNodeLabel,
                                            NewValueLabel,
                                            GraphInputName,
                                            SlotName>
    rewrite_labelled_open_kwarg_dataflow_graph_labels(
        LabelledOpenKwargDataflowGraphView<NodeLabel,
                                           ValueLabel,
                                           GraphInputName,
                                           SlotName> const &,
        F);

} // namespace FlexFlow
