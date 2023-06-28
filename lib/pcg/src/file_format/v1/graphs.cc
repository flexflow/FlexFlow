#include "pcg/file_format/v1/graphs.h"

namespace FlexFlow {

template <typename NodeT, typename TensorT>
static V1JsonableGraph<NodeT, TensorT> to_v1(OutputLabelledMultiDiGraph<NodeT, TensorT> const &g) {
  size_t node_idx = 0;   
  size_t tensor_idx = 0;


  bidict<size_t, Node> nodes = enumerate(get_nodes(g));
  bidict<size_t, NodePort> node_ports = enumerate(get_node_ports(g));

  std::unordered_map<size_t, NodeT> v1_nodes = map_values(nodes, [&](Node const &n) { return g.at(n); });
  std::unordered_map<size_t, TensorT> v1_tensors = map_values(tensors, [&]();
}

}
