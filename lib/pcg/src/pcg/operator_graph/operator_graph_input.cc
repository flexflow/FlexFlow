#include "pcg/operator_graph/operator_graph_input.h"

namespace FlexFlow {

Node get_node(OperatorGraphInput const &i) {
  return i.node;
}

int get_idx(OperatorGraphInput const &i) {
  return i.idx;
}

} // namespace FlexFlow
