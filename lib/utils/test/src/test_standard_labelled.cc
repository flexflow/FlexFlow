// #include "graph/labelled/standard_labelled.h"
// #include "test/utils/all.h"
// #include "test/utils/doctest.h"
// #include "utils/containers.h"

// #include <string>
// #include <vector>

// using namespace FlexFlow;

// TEST_CASE("LabelledMultiDiGraph  implementation") {
//   LabelledMultiDiGraph g =
//       LabelledMultiDiGraph<std::string, std::string>::create<
//           AdjacencyMultiDiGraph,
//           UnorderedLabelling<Node, std::string>,
//           UnorderedLabelling<MultiDiEdge, std::string>>();

//   int num_nodes = 3;
//   std::vector<std::string> nodel_labels = repeat(
//       num_nodes, [&](int i) { return "nodel_labels_" + std::to_string(i); });

//   std::vector<NodePort> node_ports =
//       repeat(num_nodes, [&] { return g.add_node_port(); });
//   std::vector<Node> nodes =
//       repeat(num_nodes, [&](int i) { return g.add_node(nodel_labels[i]); });

//   std::vector<NodePort> get_nodeports =
//       repeat(num_nodes, [&](int i) { return g.at(nodes[i]); });

//   CHECK(get_nodeports == node_ports);

//   std::vector<std::string> edge_labels = repeat(
//       num_nodes, [&](int i) { return "edge_labels_" + std::to_string(i); });

//   //(no,po,n1, p1), (n1,p1, n2, p2) , (n1,p1, n3, p3) this may have some
//   //problem, we can fix
//   std::vector<MultiDiEdge> multi_diedges = {
//       {nodes[0], node_ports[0], nodes[1], node_ports[1]},
//       {nodes[1], node_ports[1], nodes[2], node_ports[2]},
//       {nodess[1], node_ports[1], nodes[3], nodde_ports[3]}};

//   for (MultiDiEdge const &edge : multi_diedges) {
//     OpenMultiDiEdge e{edge};
//     g.add_edge(e);
//   }

//   repeat(num_nodes, [&](int i) { return g.add_label(e[i], edge_labels[i]); });

//   std::vector<std::string> expected_edge_labels =
//       repeat(num_nodes, [&](int i) { return g.at(e[i]); });

//   CHECK(expected_edge_labels == edge_labels);

//   CHECK(g.query_nodes(NodeQuery::all()) == without_order(nodes));

//   CHECK(
//       g.query_edges(OpenMultiDiEdgeQuery(MultiDiEdgeQuery::all())) ==
//       without_order(
//           multi_diedges)); // this may have some problem
//                            //  add test for
//                            //  MultiDiEdgeQuery::with_src_nodes/with_dst_nodes/
//   // with_src_idxs/with_dst_idxs
// }
