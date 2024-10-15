#include "test/utils/rapidcheck.h"
#include "test/utils/rapidcheck/visitable.h"
#include "utils/commutative_pair.h"
#include "utils/containers/repeat.h"
#include "utils/graph/instances/hashmap_undirected_graph.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/undirected/undirected_edge_query.h"
#include "utils/graph/undirected/undirected_graph.h"

using namespace FlexFlow;

using namespace rc;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE(
      "UndirectedGraph implementations", T, HashmapUndirectedGraph) {

    RC_SUBCASE("Full", [&]() {
      UndirectedGraph g = UndirectedGraph::create<T>();
      int num_nodes = *gen::inRange(1, 10);
      std::vector<Node> n = repeat(num_nodes, [&] { return g.add_node(); });
      int num_edges = *gen::inRange(0, num_nodes);
      std::vector<UndirectedEdge> e;
      if (num_nodes > 0) {
        e = *gen::unique<std::vector<UndirectedEdge>>(
            num_edges,
            gen::construct<UndirectedEdge>(
                gen::construct<commutative_pair<Node>>(gen::elementOf(n),
                                                       gen::elementOf(n))));
      }
      for (UndirectedEdge const &edge : e) {
        g.add_edge(edge);
      }

      CHECK(g.query_nodes(node_query_all()) == unordered_set_of(n));

      auto subset = *rc::subset_of(n);
      CHECK(g.query_nodes(NodeQuery{query_set<Node>{subset}}) == subset);

      CHECK(g.query_edges(undirected_edge_query_all()) == unordered_set_of(e));
    });
  }
}
/* static_assert(is_fmtable<UndirectedEdgeQuery>::value, ""); */

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE(
      "UndirectedGraph implementations", T, HashmapUndirectedGraph) {

    RC_SUBCASE("Full", [&]() {
      UndirectedGraph g = UndirectedGraph::create<T>();
      int num_nodes = *gen::inRange(1, 10);
      std::vector<Node> n = repeat(num_nodes, [&] { return g.add_node(); });
      int num_edges = *gen::inRange(0, num_nodes);
      std::vector<UndirectedEdge> e;
      if (num_nodes > 0) {
        e = *gen::unique<std::vector<UndirectedEdge>>(
            num_edges,
            gen::construct<UndirectedEdge>(
                gen::construct<commutative_pair<Node>>(gen::elementOf(n),
                                                       gen::elementOf(n))));
      }
      for (UndirectedEdge const &edge : e) {
        g.add_edge(edge);
      }

      CHECK(g.query_nodes(node_query_all()) == unordered_set_of(n));

      auto subset = *rc::subset_of(n);
      CHECK(g.query_nodes(NodeQuery{query_set<Node>{subset}}) == subset);

      CHECK(g.query_edges(undirected_edge_query_all()) == unordered_set_of(e));
    });
  }
}
