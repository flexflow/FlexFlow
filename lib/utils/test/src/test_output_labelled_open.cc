#include "test/utils/all.h"
#include "utils/containers.h"
#include "utils/graph/labelled/output_labelled_open.h"
#include "utils/graph/labelled/unordered_label.h"
#include "utils/graph/node.h"

#include <string>
#include <vector>

using namespace FlexFlow;

TEST_CASE("OutputLabelledOpenMultiDiGraph implementation") {
  OutputLabelledOpenMultiDiGraph g =
      OutputLabelledOpenMultiDiGraph<std::string, std::string>::create<
          AdjacencyOpenMultiDiGraph,
          UnorderedLabelling<Node, std::string>,
          UnorderedLabelling<InputMultiDiEdge, std::string>,
          UnorderedLabelling<MultiDiOutput, std::string>>();

  int num_nodes = 3;
  std::vector<std::string> nodel_labels = repeat2(
      num_nodes,
      [&](int i) { return "nodel_labels_" + std::to_string(i); },
      std::string());

  std::vector<std::string> input_edge_labels = repeat2(
      num_nodes,
      [&](int i) { return "input_edge_labels_" + std::to_string(i); },
      std::string());

  std::vector<std::string> output_edge_labels = repeat2(
      num_nodes,
      [&](int i) { return "output_edge_labels_" + std::to_string(i); },
      std::string());

  std::vector<NodePort> node_ports =
      repeat(num_nodes, [&] { return g.add_node_port(); });

  std::vector<Node> nodes;
  for (int i = 0; i < num_nodes; i++) {
    nodes.push_back(g.add_node(nodel_labels[i]));
  }

  std::vector<std::string> get_nodelabels;
  for (int i = 0; i < num_nodes; i++) {
    get_nodelabels.push_back(g.at(nodes[i]));
  }

  CHECK(get_nodelabels == nodel_labels);

  // we should add_label for input and output
  //(no,po,n1, p1), (n1,p1, n2, p2) , (n1,p1, n3, p3) this may have some
  // problem, we can fix
  std::vector<MultiDiEdge> multi_diedges = {
      {nodes[0], node_ports[0], nodes[1], node_ports[1]},
      {nodes[1], node_ports[1], nodes[2], node_ports[2]},
      {nodes[1], node_ports[1], nodes[3], node_ports[3]}};

  std::vector<InputMultiDiEdge> input_multi_diedges = {
      {dst = nodes[1], dst_idx = node_ports[1]},
      {nodes[2], node_ports[2]},
      {nodes[3], node_ports[3]}};

  std::vector<OutputMultiDiEdge> output_multi_diedges = {
      {nodes[0], node_ports[0]},
      {nodes[1], node_ports[1]},
      {nodes[1], node_ports[1]}};

  for (MultiDiEdge const &edge : multi_diedges) {
    OpenMultiDiEdge e{edge};
    g.add_edge(e);
  }

  for (int i = 0; i < input_edge_labels.size(); i++) {
    g.add_label(input_multi_diedges[i], input_edge_labels[i]);
  }

  for (int i = 0; i < output_edge_labels.size(); i++) {
    g.add_label(output_multi_diedges[i], output_edge_labels[i]);
  }

  std::vector<std::string> expected_input_edge_labels;
  for (int i = 0; i < input_edge_labels.size(); i++) {
    expected_input_edge_labels.push_back(g.at(input_multi_diedges[i]));
  }

  CHECK(expected_input_edge_labels == input_edge_labels);

  std::vector<std::string> expected_output_edge_labels;
  for (int i = 0; i < output_edge_labels.size(); i++) {
    expected_output_edge_labels.push_back(g.at(output_multi_diedges[i]));
  }

  CHECK(expected_output_edge_labels == output_edge_labels);

  CHECK(g.query_nodes(NodeQuery::all()) == nodes);

  CHECK(g.query_edges(OpenMultiDiEdgeQuery(MultiDiEdgeQuery::all())) ==
        multi_diedges); // this may have some problem
  // add test for MultiDiEdgeQuery::with_src_nodes/with_dst_nodes/
  // with_src_idxs/with_dst_idxs
}
