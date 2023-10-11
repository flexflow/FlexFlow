#ifndef UTILS_GRAPH_INCLUDE_NODE_PORT
#define UTILS_GRAPH_INCLUDE_NODE_PORT

namespace FlexFlow {

/**
 * @class NodePort
 * @brief An opaque object used to disambiguate multiple edges between the same
 * nodes in a MultiDiGraph
 *
 * Name chosen to match the terminology used by <a href="linkURL">ELK</a>
 *
 */
struct NodePort : public strong_typedef<NodePort, size_t> {
  using strong_typedef::strong_typedef;
};
FF_TYPEDEF_HASHABLE(NodePort);
FF_TYPEDEF_PRINTABLE(NodePort, "NodePort");

}


#endif
