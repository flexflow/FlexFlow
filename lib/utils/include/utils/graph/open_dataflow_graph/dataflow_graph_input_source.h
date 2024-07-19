#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_DATAFLOW_GRAPH_INPUT_SOURCE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_DATAFLOW_GRAPH_INPUT_SOURCE_H

#include "utils/graph/open_dataflow_graph/dataflow_graph_input.dtg.h"

namespace FlexFlow {

struct DataflowGraphInputSource {
public:
  DataflowGraphInputSource();

  DataflowGraphInput new_dataflow_graph_input();

private:
  static size_t next_available_uid;
};

} // namespace FlexFlow

#endif
