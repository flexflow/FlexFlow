#include "utils/graph/node/node_source.h"

namespace FlexFlow {

size_t NodeSource::next_available_node_id = 0;

NodeSource::NodeSource() {}

Node NodeSource::new_node() {
  Node result = Node{NodeSource::next_available_node_id};
  NodeSource::next_available_node_id++;
  return result;
}

} // namespace FlexFlow
