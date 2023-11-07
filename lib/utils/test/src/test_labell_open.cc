#include "test/utils/all.h"
#include "test/utils/doctest.h"
#include "utils/containers.h"
#include "utils/graph/labelled_open.h"

#include <string>

using namespace FlexFlow;

// test the LabelledOpenMultiDiGraph

TEST_CASE_TEMPLATE("LabelledOpenMultiDiGraph implementations",
                   T,
                   UnorderedLabelledOpenMultiDiGraph<int, std::string>) {
  // I define NodeLabel/ as int, EdgeLabelInputLabel/OutputLabel as string
  LabelledOpenMultiDiGraph<int, std::string> g =
      LabelledOpenMultiDiGraph<int, std::string>::create<T>();
  int num_nodes = 3;
  std::vector<Node> n =
      repeat<Node>(num_nodes, [&g](int i) { return g.add_node(i); });

  std::vector<NodePort> p =
      repeat(num_nodes, [&] { return g.add_node_port(); });

  for (int i = 0; i < num_nodes; i++) {
    CHECK(i == g.at(n[i])); // check NodeLabel &at(Node const &n);
  }

  CHECK(g.query_nodes(NodeQuery::all()) == without_order(n));

  SUBCASE("test MultiDiEdge") {
    std::vector<MultiDiEdge> edges = {
        {n[1], p[1], n[0], p[0]},
        {n[2], p[2], n[0], p[0]},
        {n[0], p[0], n[2], p[2]},
        {n[1], p[1], n[2], p[2]}}; // this may have some problem  because the
                                   // constructor for MultiDiEdge

    std::vector<std::string> edgelabels = repeat(edges.size(), [&] {
      [&](int i) { return "labels" + std::to_string(i); }
    });

    for (int i = 0; i < edges.size(); i++) {
      g.add_edge(edges[i], edgelabels[i]);
    }

    for (int i = 0; i < edges.size(); i++) {
      CHECK(edgelabels[i] == g.at(edge[i]));
    }

    OpenMultiDiEdgeQuery query{
        MultiDiEdgeQuery::all()}; // todo this may have some problem
    CHECK(g.query_edges(query) == without_order(edges));
  }

  SUBCASE("test InputMultiDiEdge") {
    std::vector<InputMultiDiEdge> edges.resize(4);
    // this may have problem to set the dst and dst_idx
    edges[0].dst = n[0];
    edges[0].dst_idx = p[0];
    edges[1].dst = n[0];
    edges[1].dst_idx = p[0];
    edges[2].dst = n[2];
    edges[2].dst_idx = p[2];
    edges[2].dst = n[2];
    edges[2].dst_idx = p[2];
    //  = {{n[1], p[1]},
    //                         {n[2], p[2]},
    //                         {n[0], p[0]},
    //                         {n[1], p[1]}};//

    std::vector<std::string> edgelabels = repeat(edges.size(), [&] {
      [&](int i) { return "labels_input_" + std::to_string(i); }
    });
    for (int i = 0; i < edges.size(); i++) {
      g.add_edge(edges[i], edgelabels[i]);
    }

    for (int i = 0; i < edges.size(); i++) {
      CHECK(edgelabels[i] == g.at(edge[i]));
    }

    OpenMultiDiEdgeQuery query(InputMultiDiEdgeQuery::all());
    CHECK(g.query_edges(query) == without_order(edges));
  }

  SUBCASE("test OutputMultiDiEdge") {
    std::vector<OutputMultiDiEdge> edges.resize(4);
    edges[0].src = n[1];
    edges[0].src_idx = p[1];
    edges[1].src = n[2];
    edges[1].src_idx = p[2];
    edges[2].src = n[0];
    edges[2].src_idx = p[0];
    edges[3].src = n[1];
    edges[3].src_idx = p[1];

    std::vector<std::string> edgelabels = repeat(edges.size(), [&] {
      [&](int i) { return "labels_output_" + std::to_string(i); }
    });

    for (int i = 0; i < edges.size(); i++) {
      g.add_edge(edges[i], edgelabels[i]);
    }

    for (int i = 0; i < edges.size(); i++) {
      CHECK(edgelabels[i] == g.at(edge[i]));
    }

    OpenMultiDiEdgeQuery query(OutputMultiDiEdgeQuery::all());
    CHECK(g.query_edges(query) == without_order(edges));
  }
}
