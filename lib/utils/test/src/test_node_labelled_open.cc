#include "test/utils/all.h"
#include "utils/containers.h"
#include "utils/graph/adjacency_openmultidigraph.h"
#include "utils/graph/labelled/node_labelled_open.h"
#include "utils/graph/labelled/unordered_label.h"
#include "utils/graph/node.h"

#include <string>
#include <vector>

using namespace FlexFlow;

// this file test the graph/labelled/node_labelled_open.h
TEST_CASE("NodeLabelledOpenMultiDiGraph implementations") {
  NodeLabelledOpenMultiDiGraph g = NodeLabelledOpenMultiDiGraph<
      std::string>::create<AdjacencyOpenMultiDiGraph,
                           UnorderedLabelling<Node, std::string>>();

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

  std::vector<std::string> expected_labels = repeat2(
      num_nodes, [&](int i) { return g.at(n[i]); }, std::string());
  CHECK(g.query_nodes(NodeQuery::all()) == without_order(n));

  for (MultiDiEdge const &edge : e) {
    g.add_edge(edge);
  }

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery{MultiDiEdgeQuery::all()}),
                  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == without_order(e));

  std::unordered_set<MultiDiEdge> res = transform(
      g.query_edges(OpenMultiDiEdgeQuery{MultiDiEdgeQuery::all().with_src_nodes(
          query_set<Node>({n[1], n[2]}))}),
      [](OpenMultiDiEdge const &edge) { return get<MultiDiEdge>(edge); });

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery{
                      MultiDiEdgeQuery::all().with_src_nodes(
                          query_set<Node>({n[1], n[2]}))}),
                  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == std::unordered_set<MultiDiEdge>{e[2], e[3]});

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery{
                      MultiDiEdgeQuery::all().with_dst_nodes(
                          query_set<Node>({n[0], n[2]}))}),
                  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == std::unordered_set<MultiDiEdge>{e[1], e[2]});

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery{
                      MultiDiEdgeQuery::all().with_src_idxs(
                          query_set<NodePort>({p[0], p[2]}))}),
                  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == without_order(e));

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery{
                      MultiDiEdgeQuery::all().with_dst_idxs(
                          query_set<NodePort>({p[0]}))}),
                  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == std::unordered_set<MultiDiEdge>{e[2]});

  CHECK(transform(g.query_edges(OpenMultiDiEdgeQuery{
                      MultiDiEdgeQuery::all()
                          .with_dst_nodes(query_set<Node>({n[1]}))
                          .with_src_nodes(query_set<Node>({n[0]}))
                          .with_src_idxs(query_set<NodePort>({p[0]}))
                          .with_dst_idxs(query_set<NodePort>({p[1]}))}),
                  [](OpenMultiDiEdge const &edge) {
                    return get<MultiDiEdge>(edge);
                  }) == std::unordered_set<MultiDiEdge>{e[0]});
}
