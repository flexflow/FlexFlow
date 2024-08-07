// #include "compiler/unity_algorithm.h"
// #include "doctest/doctest.h"
// #include "utils/graph/algorithms.h"

// using namespace FlexFlow;

// TEST_SUITE(FF_TEST_SUITE) {
//   TEST_CASE("get_source_sink_open_graph") {
//     OpenMultiDiGraph g =
//     OpenMultiDiGraph::create<AdjacencyOpenMultiDiGraph>();

//     Node n0 = g.add_node();
//     NodePort p0 = g.add_node_port();
//     InputMultiDiEdge e0{
//         n0, g.add_node_port(), std::make_pair(n0.value(), n0.value())};
//     g.add_edge(e0);

//     CHECK(bool(get_closed_sources(g) == std::unordered_set<Node>{}));
//     CHECK(bool(get_closed_sinks(g) == std::unordered_set<Node>{n0}));

//     CHECK(bool(get_open_sources(g) == std::unordered_set<Node>{n0}));
//     CHECK(bool(get_open_sinks(g) == std::unordered_set<Node>{}));
//   }

//   TEST_CASE("get_source_sink_open_graph:unconnected") {
//     OpenMultiDiGraph g =
//     OpenMultiDiGraph::create<AdjacencyOpenMultiDiGraph>();

//     Node n0 = g.add_node();
//     Node n1 = g.add_node();

//     NodePort p0 = g.add_node_port();
//     NodePort p1 = g.add_node_port();

//     InputMultiDiEdge e0{n0, p0, std::make_pair(p0.value(), p0.value())};
//     OutputMultiDiEdge e1{n1, p1, std::make_pair(p1.value(), p1.value())};
//     g.add_edge(e0);
//     g.add_edge(e1);

//     /*
//       g:  ->n0
//           n1->
//     */

//     CHECK(bool(get_closed_sources(g) == std::unordered_set<Node>{n1}));
//     CHECK(bool(get_closed_sinks(g) == std::unordered_set<Node>{n0}));

//     CHECK(bool(get_open_sources(g) == std::unordered_set<Node>{n0}));
//     CHECK(bool(get_open_sinks(g) == std::unordered_set<Node>{n1}));
//   }

//   TEST_CASE("get_cut") {
//     auto g = OpenMultiDiGraph::create<AdjacencyOpenMultiDiGraph>();

//     std::vector<Node> ns = add_nodes(g, 5);

//     MultiDiEdge e0{ns[1], g.add_node_port(), ns[0], g.add_node_port()};
//     MultiDiEdge e1{ns[2], g.add_node_port(), ns[1], g.add_node_port()};
//     MultiDiEdge e2{ns[3], g.add_node_port(), ns[1], g.add_node_port()};
//     MultiDiEdge e3{ns[4], g.add_node_port(), ns[2], g.add_node_port()};
//     MultiDiEdge e4{ns[4], g.add_node_port(), ns[3], g.add_node_port()};
//     OutputMultiDiEdge e5{
//         ns[4], g.add_node_port(), std::make_pair(ns[4].value(),
//         ns[4].value())};

//     g.add_edge(e0);
//     g.add_edge(e1);
//     g.add_edge(e2);
//     g.add_edge(e3);
//     g.add_edge(e4);
//     g.add_edge(e5);

//     GraphSplit gs0{{ns[0], ns[1]}, {ns[2], ns[3], ns[4]}};
//     CHECK(bool(get_cut_set(g, gs0) == std::unordered_set<MultiDiEdge>{e1,
//     e2}));

//     GraphSplit gs1{{ns[0], ns[1], ns[2], ns[3]}, {ns[4]}};
//     CHECK(bool(get_cut_set(g, gs1) == std::unordered_set<MultiDiEdge>{e3,
//     e4}));
//   }
// }
