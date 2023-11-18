#include "test/utils/all.h"
#include "utils/containers.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/labelled/node_labelled.h"
#include "utils/graph/labelled/unordered_label.h"
#include "utils/graph/node.h"

#include <string>
#include <vector>

using namespace FlexFlow;

TEST_CASE("NodeLabelledMultiDiGraph implementations") {
  NodeLabelledMultiDiGraph g = NodeLabelledMultiDiGraph<std::string>::
      create<AdjacencyMultiDiGraph, UnorderedLabelling<Node, std::string>>();

  int num_nodes = 3;
  std::vector<std::string> labels = repeat2(
      num_nodes,
      [&](int i) { return "labels_" + std::to_string(i); },
      std::string());

  std::vector<Node> n;
  for (int i = 0; i < num_nodes; i++) {
    n.push_back(g.add_node(labels[i]));
  }

  std::vector<NodePort> p = repeat(3, [&] { return g.add_node_port(); });

  std::vector<MultiDiEdge> e = {
      {n[1], p[1], n[0], p[0]}, // dst_node, dst_nodeport,src_node,src_nodeport,
      {n[2], p[2], n[0], p[0]},
      {n[0], p[0], n[2], p[2]},
      {n[1], p[1], n[2], p[2]}};

  for (int i = 0; i < num_nodes; i++) {
    CHECK(g.at(n[i]) == labels[i]);
  }

  CHECK(g.query_nodes(NodeQuery::all()) == without_order(n));

  for (MultiDiEdge const &edge : e) {
    g.add_edge(edge);
  }

  CHECK(g.query_edges(MultiDiEdgeQuery::all()) == without_order(e));
  // TODO: we should add more test use MultiDiEdgeQuery::with_src_nodes
}
