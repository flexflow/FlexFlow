#include "utils/graph/open_dataflow_graph/dataflow_graph_input_source.h"

namespace FlexFlow {

size_t DataflowGraphInputSource::next_available_uid = 0;

DataflowGraphInputSource::DataflowGraphInputSource() {}

DataflowGraphInput DataflowGraphInputSource::new_dataflow_graph_input() {
  DataflowGraphInput result =
      DataflowGraphInput{DataflowGraphInputSource::next_available_uid};
  DataflowGraphInputSource::next_available_uid++;
  return result;
}

} // namespace FlexFlow
