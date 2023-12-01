// #include "utils/graph/labelled/standard_labelled.h"
// #include "utils/graph/labelled/unordered_label.h"
// #include "test/utils/all.h"
// #include "utils/containers.h"
// #include "utils/graph/adjacency_multidigraph.h"
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
//   std::vector<std::string> nodel_labels = repeat2(
//       num_nodes, [&](int i) { return "nodel_labels_" + std::to_string(i); },
//       std::string());

//   std::vector<NodePort> p=
//       repeat(num_nodes, [&] { return g.add_node_port(); });
//   std::vector<Node> n;
//   for (int i = 0; i < num_nodes; i++) {
//     n.push_back(g.add_node(nodel_labels[i]));
//   }

//   std::vector<std::string> get_labels;
//   for(int i =0; i < num_nodes; i++) {
//     get_labels.push_back(g.at(n[i]));
//   }
//       //repeat(num_nodes, [&](int i) { return g.at(nodes[i]); });

//   CHECK(get_labels ==nodel_labels );

//   std::vector<std::string> edge_labels = repeat2(
//       num_nodes, [&](int i) { return "edge_labels_" + std::to_string(i); },
//       std::string());

//   //(no,po,n1, p1), (n1,p1, n2, p2) , (n1,p1, n3, p3) this may have some
//   //problem, we can fix
//   std::vector<MultiDiEdge> e = {
//       {n[1], p[1], n[0], p[0]}, // dst_node,
//       dst_nodeport,src_node,src_nodeport, {n[2], p[2], n[0], p[0]}, {n[0],
//       p[0], n[2], p[2]}, {n[1], p[1], n[2], p[2]}};

//   for (MultiDiEdge const &edge : e) {
//     g.add_edge(edge);
//   }

// //   CHECK(g.query_nodes(NodeQuery::all()) == without_order(nodes));

// //   CHECK(
// //       g.query_edges(OpenMultiDiEdgeQuery(MultiDiEdgeQuery::all())) ==
// //       without_order(
// //           multi_diedges)); // this may have some problem
// //                            //  add test for
// //                            //
// // MultiDiEdgeQuery::with_src_nodes/with_dst_nodes/
//   // with_src_idxs/with_dst_idxs
// }
