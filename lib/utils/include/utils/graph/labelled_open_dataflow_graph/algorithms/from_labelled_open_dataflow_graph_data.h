#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FROM_LABELLED_OPEN_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FROM_LABELLED_OPEN_DATAFLOW_GRAPH_DATA_H

#include "utils/containers/filtrans.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/labelled_open_dataflow_graph_data.dtg.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/algorithms/from_open_dataflow_graph_data.h"
#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_data.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
LabelledOpenDataflowGraphView<NodeLabel, ValueLabel>
    from_labelled_open_dataflow_graph_data(
        LabelledOpenDataflowGraphData<NodeLabel, ValueLabel> const &data) {
  std::unordered_set<OpenDataflowValue> values = keys(data.value_data);
  std::unordered_set<DataflowOutput> outputs =
      filtrans(values, try_get_dataflow_output);

  OpenDataflowGraphData unlabelled_data = OpenDataflowGraphData{
      keys(data.node_data),
      data.edges,
      data.inputs,
      outputs,
  };

  return with_labelling(from_open_dataflow_graph_data(unlabelled_data),
                        data.node_data,
                        data.value_data);
}

} // namespace FlexFlow

#endif
