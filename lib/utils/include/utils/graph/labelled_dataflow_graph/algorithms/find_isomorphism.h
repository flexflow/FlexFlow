#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H

#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph_view.h"
#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/dataflow_graph/algorithms/find_isomorphisms.h"
#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/view_as_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/find_isomorphism.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
std::optional<DataflowGraphIsomorphism> find_isomorphism(
    LabelledDataflowGraphView<NodeLabel, ValueLabel> const &src,
    LabelledDataflowGraphView<NodeLabel, ValueLabel> const &dst) {
  std::optional<OpenDataflowGraphIsomorphism> open_isomorphism = find_isomorphism(
    view_as_labelled_open_dataflow_graph(src),
    view_as_labelled_open_dataflow_graph(dst));

  return transform(open_isomorphism, [](OpenDataflowGraphIsomorphism const &open) {
    assert (open.input_mapping.empty());
    return DataflowGraphIsomorphism{open.node_mapping};
  });
}

} // namespace FlexFlow

#endif
