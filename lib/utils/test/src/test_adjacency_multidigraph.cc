#include "doctest.h"
#include "utils/graph/adjacency_multidigraph.h"

using namespace FlexFlow;

TEST_CASE("AdjacencyMultiDiGraph:basic_test") {
  AdjacencyMultiDiGraph g;
  Node n0 = g.add_node();
  Node n1 = g.add_node();
  Node n2 = g.add_node();
  NodePort p0(0);
  NodePort p1(1);
  NodePort p2(2);    
  MultiDiEdge e1{n0, n1, p0, p1};
  MultiDiEdge e2{n0, n2, p0, p2};
  MultiDiEdge e3{n2, n0, p2, p0};
  MultiDiEdge e4{n2, n1, p2, p1};
  g.add_edge(e1);
  g.add_edge(e2);
  g.add_edge(e3);
  g.add_edge(e4);

  CHECK(g.query_nodes({}) == std::unordered_set<Node>{n0, n1, n2});

  CHECK(g.query_nodes(NodeQuery{{n0, n2}}) == std::unordered_set<Node>{n0, n2});

  CHECK(g.query_edges({}) ==
        std::unordered_set<MultiDiEdge>{e1, e2, e3, e4});

  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_node(n1)) ==
       std::unordered_set<MultiDiEdge>{});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_node(n1)) ==
        std::unordered_set<MultiDiEdge>{e1, e4});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idx(p1)) ==
        std::unordered_set<MultiDiEdge>{});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idx(p1)) ==
        std::unordered_set<MultiDiEdge>{e1, e4});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_nodes({n1, n2})) ==
        std::unordered_set<MultiDiEdge>{e3, e4});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes({n0, n2})) ==
        std::unordered_set<MultiDiEdge>{e2, e3});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idxs({p1, p2})) ==
        std::unordered_set<MultiDiEdge>{e3, e4});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs({p0, p2})) ==
        std::unordered_set<MultiDiEdge>{e2, e3});
  CHECK(g.query_edges(MultiDiEdgeQuery::all()
                          .with_src_node(n1)
                          .with_dst_node(n2)
                          .with_src_idx(p1)
                          .with_dst_idx(p2)) ==
        std::unordered_set<MultiDiEdge>{});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idx(p2)) ==
        std::unordered_set<MultiDiEdge>{e2});
}