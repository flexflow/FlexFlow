#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"

namespace FlexFlow {

parallel_layer_guid_t get_source_layer(parallel_tensor_guid_t const &t) {
  return parallel_layer_guid_t{t.raw_graph_output.node};
}

} // namespace FlexFlow
