#include "compiler/unity_algorithm.h"
#include "doctest.h"

using namespace FlexFlow;

TEST_CASE("get_subgraph_labelled_open_graph") {
  auto g = LabelledOpenMultiDiGraph<int, int>::create<
      UnorderedLabelledOpenMultiDiGraph<int, int>>();

  int t0 = 100000;

  Node n0 = g.add_node(0);
  Node n1 = g.add_node(1);
  Node n2 = g.add_node(2);
  Node n3 = g.add_node(3);
  Node n4 = g.add_node(4);

  MultiDiEdge e0(n0, n1, 0, 0);
  MultiDiEdge e1(n0, n2, 1, 0);
  MultiDiEdge e2(n1, n3, 0, 0);
  MultiDiEdge e3(n2, n3, 0, 1);
  MultiDiEdge e4(n3, n4, 0, 0);
  OutputMultiDiEdge e5({n4.value(), t0}, n4, 0);

  g.add_edge(e0, 0);
  g.add_edge(e1, 1);
  g.add_edge(e2, 2);
  g.add_edge(e3, 3);
  g.add_edge(e4, 4);
  g.add_edge(e5, 5);

  auto subgraph0 = get_subgraph(g,
                                std::unordered_set<Node>{n3, n4},
                                InputSettings::INCLUDE_INPUTS,
                                OutputSettings::INCLUDE_OUTPUTS);
  auto subgraph1 = get_subgraph(g,
                                std::unordered_set<Node>{n3, n4},
                                InputSettings::INCLUDE_INPUTS,
                                OutputSettings::EXCLUDE_OUTPUTS);
  auto subgraph2 = get_subgraph(g,
                                std::unordered_set<Node>{n3, n4},
                                InputSettings::EXCLUDE_INPUTS,
                                OutputSettings::INCLUDE_OUTPUTS);
  auto subgraph3 = get_subgraph(g,
                                std::unordered_set<Node>{n3, n4},
                                InputSettings::EXCLUDE_INPUTS,
                                OutputSettings::EXCLUDE_OUTPUTS);

  CHECK(get_nodes(subgraph0) == std::unordered_set<Node>{n3, n4});
  CHECK(get_nodes(subgraph1) == std::unordered_set<Node>{n3, n4});
  CHECK(get_nodes(subgraph2) == std::unordered_set<Node>{n3, n4});
  CHECK(get_nodes(subgraph3) == std::unordered_set<Node>{n3, n4});

  std::unordered_set<InputMultiDiEdge> input_set{split_edge(e2).second,
                                                 split_edge(e3).second};
  std::unordered_set<OutputMultiDiEdge> output_set{e5};

  CHECK(get_inputs(subgraph0) == input_set);
  CHECK(get_inputs(subgraph1) == input_set);
  CHECK(get_inputs(subgraph2).empty());
  CHECK(get_inputs(subgraph3).empty());

  CHECK(get_outputs(subgraph0) == output_set);
  CHECK(get_outputs(subgraph1).empty());
  CHECK(get_outputs(subgraph2) == output_set);
  CHECK(get_outputs(subgraph3).empty());

  CHECK(get_edges(subgraph0) ==
        std::unordered_set<OpenMultiDiEdge>{
            split_edge(e2).second, split_edge(e3).second, e4, e5});
  CHECK(get_edges(subgraph1) ==
        std::unordered_set<OpenMultiDiEdge>{
            split_edge(e2).second, split_edge(e3).second, e4});
  CHECK(get_edges(subgraph2) == std::unordered_set<OpenMultiDiEdge>{e4, e5});
  CHECK(get_edges(subgraph3) == std::unordered_set<OpenMultiDiEdge>{e4});
}
