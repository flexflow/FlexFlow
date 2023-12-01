#include "test/utils/all.h"
#include "utils/containers.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/labelled/output_labelled.h"
#include "utils/graph/labelled/unordered_label.h"

#include <string>
#include <vector>

using namespace FlexFlow;

TEST_CASE("OutputLabelledMultiDiGraph implementation") {
  OutputLabelledMultiDiGraph g =
      OutputLabelledMultiDiGraph<std::string, std::string>::create<
          AdjacencyMultiDiGraph,
          UnorderedLabelling<Node, std::string>,
          UnorderedLabelling<MultiDiOutput, std::string>>();

  int num_nodes = 3;
  std::vector<std::string> nodel_labels = repeat2(
      num_nodes,
      [&](int i) { return "nodel_labels_" + std::to_string(i); },
      std::string());

  std::vector<std::string> output_edge_labels = repeat2(
      num_nodes,
      [&](int i) { return "output_edge_labels_" + std::to_string(i); },
      std::string());

  std::vector<NodePort> p =
      repeat(num_nodes, [&] { return g.add_node_port(); });

  std::vector<Node> n;
  for (int i = 0; i < num_nodes; i++) {
    n.push_back(g.add_node(nodel_labels[i]));
  }

  std::vector<std::string> expected_node_labels;
  for (int i = 0; i < num_nodes; i++) {
    expected_node_labels.push_back(g.at(n[i]));
  }

  CHECK(expected_node_labels == nodel_labels);

  std::vector<std::string> output_labels = repeat2(
      num_nodes,
      [&](int i) { return "output_labels_" + std::to_string(i); },
      std::string());

  //(no,po,n1, p1), (n1,p1, n2, p2) , (n1,p1, n3, p3) this may have some
  // problem, we can fix
  std::vector<MultiDiEdge> e = {{n[0], p[0], n[1], p[1]},
                                {n[1], p[1], n[2], p[2]},
                                {n[1], p[1], n[3], p[3]}};

  for (MultiDiEdge const &edge : e) {
    g.add_edge(edge);
  }

  std::vector<MultiDiOutput> multi_di_output = {
      {n[0], p[0]}, {n[1], p[1]}, {n[1], p[1]}};

  for (int i = 0; i < output_labels.size(); i++) {
    g.add_edge(multi_di_output[i], output_labels[i]);
  }

  std::vector<std::string> expected_output_labels;
  for (int i = 0; i < output_labels.size(); i++) {
    expected_output_labels.push_back(g.at(multi_di_output[i]));
  }

  CHECK(output_labels == expected_output_labels);

  CHECK(g.query_nodes(NodeQuery::all()) == without_order(n));

  //   CHECK(g.query_edges(OpenMultiDiEdgeQuery(MultiDiEdgeQuery::all())) ==
  //         without_order(multi_diedges)); // this may have some problem
  // add test for MultiDiEdgeQuery::with_src_nodes/with_dst_nodes/
  // with_src_idxs/with_dst_idxs
}
