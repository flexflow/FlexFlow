#include "pcg/operator_graph/operator_graph_output.h"

namespace FlexFlow {

Node get_node(OperatorGraphOutput const &o) {
  return o.node;
}

int get_idx(OperatorGraphOutput const &o) {
  return o.idx;
}

} // namespace FlexFlow
