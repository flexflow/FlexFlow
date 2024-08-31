#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_IS_ISOMORPHIC_UNDER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_IS_ISOMORPHIC_UNDER_H

#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.dtg.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
bool is_isomorphic_under(
    LabelledDataflowGraphView<NodeLabel, ValueLabel> const &src, 
    LabelledDataflowGraphView<NodeLabel, ValueLabel> const &dst,
    DataflowGraphIsomorphism const &candidate_isomorphism) {
  return is_isomorphic_under(
    view_as_labelled_open_dataflow_graph(src),
    view_as_labelled_open_dataflow_graph(dst),
    OpenDataflowGraphIsomorphism{
      candidate_isomorphism.node_mapping,
      {},
    }
  );
}

} // namespace FlexFlow

#endif
