#include "graph/labelled/output_labelled_open.h"
#include "test/utils/all.h"
#include "test/utils/doctest.h"
#include "utils/containers.h"

using namespace FlexFlow;

TEST_CASE("OutputLabelledOpenMultiDiGraph implementation") {
  OutputLabelledOpenMultiDiGraph g =
      OutputLabelledOpenMultiDiGraph<std::string, std::string>::create<
          OutputLabelledOpenMultiDiGraph,
          UnorderedLabelling<Node, std::string>,
          UnorderedLabelling<InputMultiDiEdge, std::string>,
          UnorderedLabelling<MultiDiOutput, std::string>>();

  int num_nodes = 3;
  std::vector<std::string> nodel_labels = repeat(
      num_nodes, [&](int i) { return "nodel_labels_" + std::to_string(i); });

  std::vector<std::string> input_edge_labels = repeat(num_nodes, [&](int i) {
    return "input_edge_labels_" + std::to_string(i);
  });
  std::vector<std::string> output_edge_labels = repeat(num_nodes, [&](int i) {
    return "output_edge_labels_" + std::to_string(i);
  });

  std::vector<NodePort> node_ports =
      repeat(num_nodes, [&] { return g.add_node_port(); });

  std::vector<Node> nodes =
      repeat(num_nodes, [&](NodePort p) { return g.add_node(p); });

  std::vector<NodePort> get_nodeports =
      repeat(num_nodes, [&](Node n) { return g.at(n); });

  CHECK(get_nodeports == node_ports);

  // we should add_label for input and output
  //(no,po,n1, p1), (n1,p1, n2, p2) , (n1,p1, n3, p3) this may have some
  //problem, we can fix
  std::vector<MultiDiEdge> multi_diedges = {
      {nodes[0], node_ports[0], nodes[1], node_ports[1]},
      {nodes[1], node_ports[1], nodes[2], node_ports[2]},
      {nodess[1], node_ports[1], nodes[3], nodde_ports[3]}};

  std::vector<InputMultiDiEdge> input_multi_diedges = {
      {nodes[1], node_ports[1]},
      {nodes[2], node_ports[2]},
      {nodes[3], nodde_ports[3]}};

  std::vector<OutputMultiDiEdge> output_multi_diedges = {
      {nodes[0], node_ports[0]},
      {nodes[1], node_ports[1]},
      {nodess[1], node_ports[1]}};

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
