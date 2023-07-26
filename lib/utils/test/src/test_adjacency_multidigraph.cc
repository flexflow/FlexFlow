#include "doctest.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/multidigraph_interfaces.h"

using namespace FlexFlow;

TEST_CASE("AdjacencyMultiDiGraph:basic_test") {
  MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();

  std::vector<Node> nodes = g.add_nodes(3);
  std::vector<NodePort> ports = g.add_node_ports(3);

  Node n0 = nodes[0];
  Node n1 = nodes[1];
  Node n2 = nodes[2];
  NodePort p0 = ports[0];
  NodePort p1 = ports[1];
  NodePort p2 = ports[2];
  MultiDiEdge e1{n0, n1, p0, p1};
  MultiDiEdge e2{n0, n2, p0, p2};
  MultiDiEdge e3{n2, n0, p2, p0};
  MultiDiEdge e4{n2, n1, p2, p1};

  std::vector<MultiDiEdge> edges = {e1, e2, e3, e4};
  g.add_edges(edges);

  CHECK(g.query_nodes(NodeQuery::all()) ==
        std::unordered_set<Node>{n0, n1, n2});

  CHECK(g.query_nodes(NodeQuery(query_set<Node>({n0, n2}))) ==
        std::unordered_set<Node>{n0, n2});

  CHECK(g.query_edges(MultiDiEdgeQuery::all()) ==
        std::unordered_set<MultiDiEdge>{e1, e2, e3, e4});

  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_nodes({n1})) ==
        std::unordered_set<MultiDiEdge>{});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes({n1})) ==
        std::unordered_set<MultiDiEdge>{e1, e4});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idxs({p1})) ==
        std::unordered_set<MultiDiEdge>{});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs({p1})) ==
        std::unordered_set<MultiDiEdge>{e1, e4});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_nodes(query_set<Node>(
            {n1, n2}))) == std::unordered_set<MultiDiEdge>{e3, e4});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes(query_set<Node>(
            {n0, n2}))) == std::unordered_set<MultiDiEdge>{e2, e3});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idxs(query_set<NodePort>(
            {p1, p2}))) == std::unordered_set<MultiDiEdge>{e3, e4});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs(query_set<NodePort>(
            {p0, p2}))) == std::unordered_set<MultiDiEdge>{e2, e3});
  CHECK(g.query_edges(MultiDiEdgeQuery::all()
                          .with_src_nodes({n1})
                          .with_dst_nodes({n2})
                          .with_src_idxs({p1})
                          .with_dst_idxs({p2})) ==
        std::unordered_set<MultiDiEdge>{});
  CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs({p2})) ==
        std::unordered_set<MultiDiEdge>{e2});

  SUBCASE("remove node") {
    g.remove_node_unsafe(n0);

    CHECK(g.query_nodes(NodeQuery::all()) == std::unordered_set<Node>{n1, n2});

    CHECK(g.query_edges(MultiDiEdgeQuery::all()) ==
          std::unordered_set<MultiDiEdge>{e3, e4});

    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_nodes({n0})) ==
          std::unordered_set<MultiDiEdge>{});

    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes({n0})) ==
          std::unordered_set<MultiDiEdge>{e3});

    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idxs({p2})) ==
          std::unordered_set<MultiDiEdge>{e3, e4});
    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs({p0})) ==
          std::unordered_set<MultiDiEdge>{e3});
  }

  SUBCASE("remove_edge") {
    g.remove_edge(e1);

    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_nodes({n0}).with_dst_nodes(
              {n1})) == std::unordered_set<MultiDiEdge>{});

    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes({n2})) ==
          std::unordered_set<MultiDiEdge>{e2});

    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idxs({p2})) ==
          std::unordered_set<MultiDiEdge>{e3, e4});
  }
}
