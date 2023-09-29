#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_DIEDGE
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_DIEDGE

#include "node.h"
#include "query_set.h"

namespace FlexFlow {

struct DiInput {
    Node dst;
};
FF_VISITABLE_STRUCT(DiInput, dst);
FF_VISIT_FMTABLE(DiInput);

struct DiOutput {
    Node src;
};
FF_VISITABLE_STRUCT(DiOutput, src);
FF_VISIT_FMTABLE(DiOutput);

struct DirectedEdge : virtual DiInput, virtual DiOutput {
  Node src;
  Node dst;
};
FF_VISITABLE_STRUCT(DirectedEdge, src, dst);
FF_VISIT_FMTABLE(DirectedEdge);

struct DirectedEdgeQuery {
  query_set<Node> srcs;
  query_set<Node> dsts;

  static DirectedEdgeQuery all();
};
FF_VISITABLE_STRUCT(DirectedEdgeQuery, srcs, dsts);
FF_VISIT_FMTABLE(DirectedEdgeQuery);

bool matches_edge(DirectedEdgeQuery const &, DirectedEdge const &);

DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &,
                                     DirectedEdgeQuery const &);

}

#endif
