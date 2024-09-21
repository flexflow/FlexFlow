#include "utils/graph/node/algorithms/generate_new_node_id_permutation.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_source.h"

namespace FlexFlow {

bidict<NewNode, Node> generate_new_node_id_permutation(GraphView const &g) {
  NodeSource node_source;
  return generate_bidict(
             get_nodes(g),
             [&](Node const &) { return NewNode{node_source.new_node()}; })
      .reversed();
}

} // namespace FlexFlow
