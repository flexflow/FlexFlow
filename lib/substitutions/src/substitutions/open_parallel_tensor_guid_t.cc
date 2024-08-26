#include "substitutions/open_parallel_tensor_guid_t.h"

namespace FlexFlow {

open_parallel_tensor_guid_t
    open_parallel_tensor_guid_from_closed(parallel_tensor_guid_t t) {
  return open_parallel_tensor_guid_t{OpenDataflowValue{t.raw_graph_output}};
}

open_parallel_tensor_guid_t
    open_parallel_tensor_guid_from_input(input_parallel_tensor_guid_t i) {
  return open_parallel_tensor_guid_t{
      OpenDataflowValue{i.raw_dataflow_graph_input}};
}

} // namespace FlexFlow
