#include "doctest.h"
#include "utils/graph/adjacency_multidigraph.h"

using namespace FlexFlow;

TEST_CASE("AdjacencyMultiDiGraph:basic_test") {
  AdjacencyMultiDiGraph g;
  Node n1 = g.add_node();
  Node n2 = g.add_node();
  Node n3 = g.add_node();
  NodePort p1 = g.add_node_port();
  NodePort p2 = g.add_node_port();
  NodePort p3 = g.add_node_port();
  NodePort p4 = g.add_node_port();
  MultiDiEdge e1{n1, n2, p1, p1};
  MultiDiEdge e2{n1, n3, p1, p2};
  MultiDiEdge e3{n3, n1, p2, p2};
  MultiDiEdge e4{n3, n2, p2, p2};
  MultiDiEdge e5{n2, n2, p3, p3};
  g.add_edge(e1);
  g.add_edge(e2);
  g.add_edge(e3);
  g.add_edge(e4);
  g.add_edge(e5);

  CHECK(g.query_nodes(query_set<Node>::matchall()) == std::unordered_set<Node>{n1, n2, n3});

  CHECK(g.query_nodes(NodeQuery{{n1, n3}}) == std::unordered_set<Node>{n1, n3});

  CHECK(g.query_edges(MultiDiEdgeQuery::all()) ==
        std::unordered_set<MultiDiEdge>{e1, e2, e3, e4, e5});

  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_nodes(n1)) ==
        std::unordered_set<MultiDiEdge>{e1, e2});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes(n1)) ==
        std::unordered_set<MultiDiEdge>{e3});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idxs(p2)) ==
        std::unordered_set<MultiDiEdge>{e3, e4});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs(p2)) ==
        std::unordered_set<MultiDiEdge>{e2, e3, e4});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_nodes({n1, n2})) ==
        std::unordered_set<MultiDiEdge>{e1, e2, e5});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes({n1, n3})) ==
        std::unordered_set<MultiDiEdge>{e2, e3});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idxs({p2, p3})) ==
        std::unordered_set<MultiDiEdge>{e3, e4, e5});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs({p1, p3})) ==
        std::unordered_set<MultiDiEdge>{e1, e5});
  CHECK(g.query_edges(MultiDiEdgeQuery::all()
                          .with_src_nodes(n1)
                          .with_dst_nodes(n3)
                          .with_src_idxs(p1)
                          .with_dst_idxs(p2)) ==
        std::unordered_set<MultiDiEdge>{e2});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs(p4)) ==
        std::unordered_set<MultiDiEdge>{});
}
