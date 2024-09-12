#include "utils/graph/undirected/undirected_edge.h"
#include "utils/hash/tuple.h"
#include <sstream>

namespace FlexFlow {

bool is_connected_to(UndirectedEdge const &e, Node const &n) {
  return e.endpoints.min() == n || e.endpoints.max() == n;
}

} // namespace FlexFlow
