// #include "compiler/unity_algorithm.h"
// #include "doctest/doctest.h"
// // #include "rapidcheck.h"

// using namespace FlexFlow;

// TEST_SUITE(FF_TEST_SUITE) {
//   TEST_CASE("get_subgraph(OpenMultiDiGraphView)") {
//     auto g = OpenMultiDiGraph::create<AdjacencyOpenMultiDiGraph>();

//     Node n0 = g.add_node();
//     Node n1 = g.add_node();
//     Node n2 = g.add_node();
//     Node n3 = g.add_node();
//     Node n4 = g.add_node();

//     NodePort p0 = g.add_node_port();
//     NodePort p1 = g.add_node_port();
//     NodePort p2 = g.add_node_port();
//     NodePort p3 = g.add_node_port();
//     NodePort p4 = g.add_node_port();
//     NodePort p5 = g.add_node_port();
//     NodePort p6 = g.add_node_port();
//     NodePort p7 = g.add_node_port();
//     NodePort p8 = g.add_node_port();
//     NodePort p9 = g.add_node_port();

//     MultiDiEdge e0{n1, p1, n0, p0};
//     MultiDiEdge e1{n2, p2, n0, p0};
//     MultiDiEdge e2{n3, p5, n1, p3};
//     MultiDiEdge e3{n3, p6, n2, p4};
//     MultiDiEdge e4{n4, p8, n3, p7};
//     OutputMultiDiEdge e5{n4, p9, std::make_pair(p9.value(), p9.value())};

//     g.add_edge(e0);
//     g.add_edge(e1);
//     g.add_edge(e2);
//     g.add_edge(e3);
//     g.add_edge(e4);
//     g.add_edge(e5);

//     std::unordered_set node_set0{n3, n4};

//     auto subgraph0 = get_subgraph<OpenMultiDiSubgraphView>(g, node_set0);
//     auto subgraph1 = get_subgraph<UpwardOpenMultiDiSubgraphView>(g,
//     node_set0); auto subgraph2 =
//         get_subgraph<DownwardOpenMultiDiSubgraphView>(g, node_set0);
//     auto subgraph3 = get_subgraph<ClosedMultiDiSubgraphView>(g, node_set0);

//     CHECK(bool(get_nodes(subgraph0) == node_set0));
//     CHECK(bool(get_nodes(subgraph1) == node_set0));
//     CHECK(bool(get_nodes(subgraph2) == node_set0));
//     CHECK(bool(get_nodes(subgraph3) == node_set0));

//     std::unordered_set<InputMultiDiEdge> input_set{split_edge(e2).second,
//                                                    split_edge(e3).second};
//     std::unordered_set<OutputMultiDiEdge> output_set{e5};

//     CHECK(bool(get_open_inputs(subgraph0) == input_set));
//     CHECK(bool(get_open_inputs(subgraph1) == input_set));
//     CHECK(bool(get_open_inputs(subgraph2).empty()));
//     CHECK(bool(get_open_inputs(subgraph3).empty()));

//     CHECK(bool(get_open_outputs(subgraph0) == output_set));
//     CHECK(bool(get_open_outputs(subgraph1).empty()));
//     CHECK(bool(get_open_outputs(subgraph2) == output_set));
//     CHECK(bool(get_open_outputs(subgraph3).empty()));

//     CHECK(bool(get_edges(subgraph0) ==
//                std::unordered_set<OpenMultiDiEdge>{
//                    split_edge(e2).second, split_edge(e3).second, e4, e5}));
//     CHECK(bool(get_edges(subgraph1) ==
//                std::unordered_set<OpenMultiDiEdge>{
//                    split_edge(e2).second, split_edge(e3).second, e4}));
//     CHECK(bool(get_edges(subgraph2) ==
//                std::unordered_set<OpenMultiDiEdge>{e4, e5}));
//     CHECK(
//         bool(get_edges(subgraph3) ==
//         std::unordered_set<OpenMultiDiEdge>{e4}));

//     CHECK(bool(get_closed_sources(subgraph2) ==
//     std::unordered_set<Node>{n3}));
//   }

//   TEST_CASE("view OutputLabelledMultiDiGraph as open") {
//     OutputLabelledMultiDiGraph<int, int> g =
//         OutputLabelledMultiDiGraph<int, int>::create<
//             UnorderedOutputLabelledMultiDiGraph<int, int>>();

//     Node n0 = g.add_node(0);
//     Node n1 = g.add_node(1);

//     NodePort p0 = g.add_node_port();
//     NodePort p1 = g.add_node_port();

//     MultiDiEdge e0{n1, p1, n0, p0};

//     g.add_edge(e0);
//     g.add_output(e0, 2);

//     CHECK(bool(get_edges(g).size() == 1));

//     OutputLabelledOpenMultiDiGraphView<int, int> open_graph =
//         view_output_labelled_as_output_labelled_open(g);

//     CHECK(bool(open_graph.at(n0) == 0));
//     CHECK(bool(open_graph.at(n1) == 1));
//     CHECK(bool(open_graph.at(e0) == 2));

//     CHECK(get_edges(open_graph).size() == 1);
//   }

//   TEST_CASE("OutputLabelledOpenMultiDiGraph") {
//     OutputLabelledOpenMultiDiGraph<int, int> g =
//         OutputLabelledOpenMultiDiGraph<int, int>::create<
//             UnorderedOutputLabelledOpenMultiDiGraph<int, int>>();

//     Node n0 = g.add_node(0);
//     Node n1 = g.add_node(1);

//     NodePort p0 = g.add_node_port();
//     NodePort p1 = g.add_node_port();

//     MultiDiEdge e0{n1, p1, n0, p0};

//     g.add_edge(e0);
//     g.add_label(e0, 2);

//     CHECK(bool(g.query_edges(OpenMultiDiEdgeQuery::all()).size() == 1));
//     CHECK(bool(get_edges(g).size() == 1));
//   }
// }
