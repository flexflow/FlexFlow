#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_labels.h"
#include "utils/archetypes/value_type.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using NewNodeLabel = value_type<2>;
using NewValueLabel = value_type<3>;

struct F {
  NewNodeLabel operator()(Node const &, NodeLabel const &) {
    PANIC();
  }
  NewValueLabel operator()(OpenDataflowValue const &, ValueLabel) {
    PANIC();
  }
};

template LabelledOpenDataflowGraphView<NewNodeLabel, NewValueLabel>
    rewrite_labels(LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &,
                   F);

} // namespace FlexFlow
