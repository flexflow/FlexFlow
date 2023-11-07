#include "test/utils/all.h"
#include "test/utils/doctest.h"
#include "utils/containers.h"
#include "graph/labelled/node_labelled.h"

using namespace FlexFlow;

TEST_CASE("NodeLabelledMultiDiGraph implementations") { 
    NodeLabelledMultiDiGraph  g = NodeLabelledMultiDiGraph::create<AdjacencyMultiDiGraph, UnorderedLabelling<std::string, Node>>();

    int num_nodes = 3;
    std::vector<std::string> labels =
      repeat(num_nodes, [&](int i) { return "labels_" + std::to_string(i); });

    std::vector<Node> nodes;
    for (int i = 0; i < num_nodes; i++) {
        nodes.push_back(g.add_node(labels[i]));
    }

    std::vector<NodePort> p = repeat(3, [&] { return g.add_node_port(); });

    std::vector<MultiDiEdge> e = {
      {n[1], p[1], n[0], p[0]},
      {n[2], p[2], n[0], p[0]},
      {n[0], p[0], n[2], p[2]},
      {n[1], p[1], n[2], p[2]}}; // this may have problem, we can fix

    for (int i = 0; i < num_nodes; i++) {
        CHECK(g.at(node[i])) == labels[i];
    }

    CHECK(g.query_nodes(NodeQuery::all()) == without_order(nodes));

    for (MultiDiEdge const &edge : e) {
        g.add_edge(edge);
    }

    CHECK(g.query_edges(MultiDiEdgeQuery::all()) == e);
    //TODO: we should add more test use MultiDiEdgeQuery::with_src_nodes

}

