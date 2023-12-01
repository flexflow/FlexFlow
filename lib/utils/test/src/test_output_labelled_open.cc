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

  std::vector<MultiDiEdge> multi_diedges = {
      {nodes[1], node_ports[1], nodes[0], node_ports[0]}, // dst_node, dst_nodeport,src_node,src_nodeport,
      {nodes[2], node_ports[2], nodes[0], node_ports[0]},
      {nodes[0], node_ports[0], nodes[2], node_ports[2]},
      {nodes[1], node_ports[1], nodes[2], node_ports[2]}};

  for (MultiDiEdge const &edge : multi_diedges) {
    OpenMultiDiEdge e{edge};
    g.add_edge(e);
  }

  CHECK(g.query_nodes(NodeQuery::all()) == without_order(nodes));

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery(MultiDiEdgeQuery::all())),
  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == without_order(multi_diedges));

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery{
                      MultiDiEdgeQuery::all().with_src_nodes(
                          query_set<Node>({nodes[1], nodes[2]}))}),
                  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == std::unordered_set<MultiDiEdge>{multi_diedges[2], multi_diedges[3]});

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery{
                      MultiDiEdgeQuery::all().with_dst_nodes(
                          query_set<Node>({nodes[0], nodes[2]}))}),
                  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == std::unordered_set<MultiDiEdge>{multi_diedges[1], multi_diedges[2]});

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery{
                      MultiDiEdgeQuery::all().with_src_idxs(
                          query_set<NodePort>({node_ports[0], node_ports[2]}))}),
                  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == without_order(multi_diedges));

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery{
                      MultiDiEdgeQuery::all().with_dst_idxs(
                          query_set<NodePort>({node_ports[0]}))}),
                  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == std::unordered_set<MultiDiEdge>{multi_diedges[2]});

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery{
                      MultiDiEdgeQuery::all()
                          .with_dst_nodes(query_set<Node>({nodes[1]}))
                          .with_src_nodes(query_set<Node>({nodes[0]}))
                          .with_src_idxs(query_set<NodePort>({node_ports[0]}))
                          .with_dst_idxs(query_set<NodePort>({node_ports[1]}))}),
                  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == std::unordered_set<MultiDiEdge>{multi_diedges[0]});
  
}
