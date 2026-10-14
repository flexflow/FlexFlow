#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_labels.h"
#include "utils/archetypes/value_type.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using NewNodeLabel = value_type<2>;
using NewValueLabel = value_type<3>;

struct F {
  NewNodeLabel operator()(Node const &, NodeLabel const &) {
    NOT_IMPLEMENTED();
  }
  NewValueLabel operator()(OpenDataflowValue const &, ValueLabel) {
    NOT_IMPLEMENTED();
  }
};

template LabelledOpenDataflowGraphView<NewNodeLabel, NewValueLabel>
    rewrite_labels(LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &,
                   F);

} // namespace FlexFlow
