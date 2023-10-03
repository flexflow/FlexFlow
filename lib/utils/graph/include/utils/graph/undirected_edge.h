#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_UNDIRECTED_EDGE
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_UNDIRECTED_EDGE

#include "node.h"

namespace FlexFlow {

struct UndirectedEdge {
public:
  UndirectedEdge() = delete;
  UndirectedEdge(Node const &src, Node const &dst);

public:
  Node smaller;
  Node bigger;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(UndirectedEdge, smaller, bigger);
FF_VISIT_FMTABLE(UndirectedEdge);

bool is_connected_to(UndirectedEdge const &, Node const &);

struct UndirectedEdgeQuery {
  query_set<Node> nodes;

  static UndirectedEdgeQuery all();
};
FF_VISITABLE_STRUCT(UndirectedEdgeQuery, nodes);
FF_VISIT_FMTABLE(UndirectedEdgeQuery);

UndirectedEdgeQuery query_intersection(UndirectedEdgeQuery const &,
                                       UndirectedEdgeQuery const &);


}

#endif
