#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/multidiedge.h"
#include "utils/graph/multidigraph_interfaces.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("MultiDiGraph implementations", T, AdjacencyMultiDiGraph) {
    MultiDiGraph g = MultiDiGraph::create<T>();

    std::vector<Node> n = repeat(3, [&] { return g.add_node(); });
    std::vector<NodePort> p = repeat(3, [&] { return g.add_node_port(); });

    std::vector<MultiDiEdge> e = {{n[1], p[1], n[0], p[0]},
                                  {n[2], p[2], n[0], p[0]},
                                  {n[0], p[0], n[2], p[2]},
                                  {n[1], p[1], n[2], p[2]}};
    for (MultiDiEdge const &edge : e) {
      g.add_edge(edge);
    }

    CHECK(g.query_nodes(NodeQuery::all()) ==
          std::unordered_set<Node>{n[0], n[1], n[2]});

    CHECK(g.query_nodes(NodeQuery{query_set<Node>{{n[0], n[2]}}}) ==
          std::unordered_set<Node>{n[0], n[2]});

    CHECK(g.query_edges(MultiDiEdgeQuery::all()) ==
          std::unordered_set<MultiDiEdge>{e[0], e[1], e[2], e[3]});

    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_nodes({n[1]})) ==
          std::unordered_set<MultiDiEdge>{});
    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes({n[1]})) ==
          std::unordered_set<MultiDiEdge>{e[0], e[3]});
    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idxs({p[1]})) ==
          std::unordered_set<MultiDiEdge>{});
    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs({p[1]})) ==
          std::unordered_set<MultiDiEdge>{e[0], e[3]});
    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_nodes(query_set<Node>(
              {n[1], n[2]}))) == std::unordered_set<MultiDiEdge>{e[2], e[3]});
    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes(query_set<Node>(
              {n[0], n[2]}))) == std::unordered_set<MultiDiEdge>{e[1], e[2]});
    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idxs(
              query_set<NodePort>({p[1], p[2]}))) ==
          std::unordered_set<MultiDiEdge>{e[2], e[3]});
    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs(
              query_set<NodePort>({p[0], p[2]}))) ==
          std::unordered_set<MultiDiEdge>{e[1], e[2]});
    CHECK(g.query_edges(MultiDiEdgeQuery::all()
                            .with_src_nodes({n[1]})
                            .with_dst_nodes({n[2]})
                            .with_src_idxs({p[1]})
                            .with_dst_idxs({p[2]})) ==
          std::unordered_set<MultiDiEdge>{});
    CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs({p[2]})) ==
          std::unordered_set<MultiDiEdge>{e[1]});

    SUBCASE("remove node") {
      g.remove_node_unsafe(n[0]);

      CHECK(g.query_nodes(NodeQuery::all()) ==
            std::unordered_set<Node>{n[1], n[2]});

      CHECK(g.query_edges(MultiDiEdgeQuery::all()) ==
            std::unordered_set<MultiDiEdge>{e[2], e[3]});

      CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_nodes({n[0]})) ==
            std::unordered_set<MultiDiEdge>{});

      CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes({n[0]})) ==
            std::unordered_set<MultiDiEdge>{e[2]});

      CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idxs({p[2]})) ==
            std::unordered_set<MultiDiEdge>{e[2], e[3]});
      CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_idxs({p[0]})) ==
            std::unordered_set<MultiDiEdge>{e[2]});
    }

    SUBCASE("remove_edge") {
      g.remove_edge(e[0]);

      CHECK(g.query_edges(
                MultiDiEdgeQuery::all().with_src_nodes({n[0]}).with_dst_nodes(
                    {n[1]})) == std::unordered_set<MultiDiEdge>{});

      CHECK(g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes({n[2]})) ==
            std::unordered_set<MultiDiEdge>{e[1]});

      CHECK(g.query_edges(MultiDiEdgeQuery::all().with_src_idxs({p[2]})) ==
            std::unordered_set<MultiDiEdge>{e[2], e[3]});
    }
  }
}
