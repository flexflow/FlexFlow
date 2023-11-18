// #include "test/utils/all.h"
// #include "test/utils/rapidcheck/visitable.h"
// #include "utils/containers.h"
// #include "utils/graph/open_graphs.h"
// #include "utils/graph/adjacency_openmultidigraph.h"

// #include <vector>

// using namespace FlexFlow;

// using namespace rc;

// TEST_CASE_TEMPLATE("OpenMultiDiGraph implementations",
//                    T,
//                    AdjacencyOpenMultiDiGraph) {

//   rc::dc_check("Full", [&]() {
//     OpenMultiDiGraph g = OpenMultiDiGraph::create<T>();
//     int num_nodes = *gen::inRange(1, 10);
//     std::vector<Node> n = repeat(num_nodes, [&] { return g.add_node(); });
//     int num_edges = *gen::inRange(0, num_nodes);
//     // we use MultiDiEdge as test  OpenMultiDiEdge
//     std::vector<MultiDiEdge> e;
//     if (num_nodes > 0) {
//       e = *gen::unique<std::vector<MultiDiEdge>>(
//           num_edges,
//           gen::construct<MultiDiEdge>(gen::elementOf(n), gen::elementOf(n)));
//     }
//     std::vector<OpenMultiDiEdge> open_edges;
//     for (MultiDiEdge const &edge : e) {
//       OpenMultiDiEdge open_edge = OpenMultiDiEdge(edge);
//       open_edges.push_back(open_edge);
//       g.add_edge(open_edge);
//     }

//     CHECK(g.query_nodes(NodeQuery::all()) == without_order(n));
//     auto subset = *rc::subset_of(n);
//     CHECK(g.query_nodes(NodeQuery{query_set<Node>{subset}}) == subset);

//     CHECK(g.query_edges(OpenMultiDiEdgeQuery::all()) ==
//           without_order(open_edges)); // this may be problem, because
//                                       // OpenMultiDiEdge is a variant
//   });
// }
