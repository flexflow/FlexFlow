#include "test/utils/all.h"
#include "test/utils/rapidcheck/visitable.h"
#include "utils/graph/hashmap_undirected_graph.h"
#include "utils/graph/undirected.h"

/* namespace rc { */

/* template <> */
/* struct Arbitrary<UndirectedGraph> { */
/*   static Gen<UndirectedGraph> arbitrary() { */
/*     int num_nodes = *gen::inRange( */
/*   } */
/* }; */

/* } */

using namespace FlexFlow;

using namespace rc;

/* static_assert(supports_rc_arbitrary<Node>::value, ""); */
/* static_assert(is_strong_typedef<Node>::value, ""); */
/* static_assert(supports_rc_arbitrary<std::tuple<int, int, int>>::value, "");
 */
/* static_assert(supports_rc_arbitrary<visit_as_tuple_t<UndirectedEdge>>::value,
 * ""); */
/* static_assert(supports_rc_arbitrary<UndirectedEdge>::value, ""); */
/* static_assert(is_fmtable<Node>::value, ""); */
/* static_assert(is_fmtable<UndirectedEdge>::value, ""); */
/* static_assert(is_streamable<UndirectedEdge>::value, ""); */
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
            gen::construct<UndirectedEdge>(gen::elementOf(n),
                                           gen::elementOf(n)));
      }
      for (UndirectedEdge const &edge : e) {
        g.add_edge(edge);
      }

      CHECK(g.query_nodes(NodeQuery::all()) == unordered_set_of(n));

      auto subset = *rc::subset_of(n);
      CHECK(g.query_nodes(NodeQuery{query_set<Node>{subset}}) == subset);

      CHECK(g.query_edges(UndirectedEdgeQuery::all()) == unordered_set_of(e));
    });
  }
}
