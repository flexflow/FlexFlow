#ifndef FLEXFLOW_GRAPH_GENERATION_H
#define FLEXFLOW_GRAPH_GENERATION_H

#include "distributions.h"
#include "sample_graphs.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/is_2_terminal_dag.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/serial_parallel/graph_generation.h"
#include <tuple>

namespace FlexFlow {

std::tuple<DiGraph, Node, Node> make_normal_nasnet_cell() {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> inputs = add_nodes(g, 2);
  std::vector<Node> sep = add_nodes(g, 5);
  std::vector<Node> id = add_nodes(g, 2);
  std::vector<Node> avg = add_nodes(g, 3);
  std::vector<Node> add = add_nodes(g, 5);
  std::vector<Node> concat = add_nodes(g, 1);

  std::vector<DirectedEdge> edges = {DirectedEdge{inputs.at(0), sep.at(1)},
                                     DirectedEdge{inputs.at(0), id.at(1)},
                                     DirectedEdge{inputs.at(0), avg.at(1)},
                                     DirectedEdge{inputs.at(0), avg.at(2)},
                                     DirectedEdge{inputs.at(0), sep.at(3)},
                                     DirectedEdge{inputs.at(0), sep.at(4)},
                                     DirectedEdge{inputs.at(1), sep.at(0)},
                                     DirectedEdge{inputs.at(1), id.at(0)},
                                     DirectedEdge{inputs.at(1), avg.at(0)},
                                     DirectedEdge{inputs.at(1), sep.at(2)},
                                     DirectedEdge{sep.at(0), add.at(0)},
                                     DirectedEdge{id.at(0), add.at(0)},
                                     DirectedEdge{sep.at(1), add.at(1)},
                                     DirectedEdge{sep.at(2), add.at(1)},
                                     DirectedEdge{avg.at(0), add.at(2)},
                                     DirectedEdge{id.at(1), add.at(2)},
                                     DirectedEdge{avg.at(1), add.at(3)},
                                     DirectedEdge{avg.at(2), add.at(3)},
                                     DirectedEdge{sep.at(3), add.at(4)},
                                     DirectedEdge{sep.at(4), add.at(4)}};
  add_edges(g, edges);

  for (Node const &a : add) {
    g.add_edge(DirectedEdge{a, concat.at(0)});
  }

  assert(get_sinks(g).size() == 1);
  assert(get_sources(g).size() == 2);
  assert(is_acyclic(g));
  return {g, inputs.at(0), inputs.at(1)};
}

std::tuple<DiGraph, Node, Node> make_reduction_nasnet_cell() {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> inputs = add_nodes(g, 2);
  std::vector<Node> sep = add_nodes(g, 5);
  std::vector<Node> id = add_nodes(g, 1);
  std::vector<Node> avg = add_nodes(g, 2);
  std::vector<Node> max = add_nodes(g, 2);
  std::vector<Node> add = add_nodes(g, 5);
  std::vector<Node> concat = add_nodes(g, 1);

  std::vector<DirectedEdge> edges = {DirectedEdge{inputs.at(0), sep.at(0)},
                                     DirectedEdge{inputs.at(0), sep.at(2)},
                                     DirectedEdge{inputs.at(0), sep.at(3)},
                                     DirectedEdge{inputs.at(1), max.at(1)},
                                     DirectedEdge{inputs.at(1), sep.at(1)},
                                     DirectedEdge{inputs.at(1), max.at(0)},
                                     DirectedEdge{inputs.at(1), avg.at(0)},
                                     DirectedEdge{sep.at(0), add.at(0)},
                                     DirectedEdge{sep.at(1), add.at(0)},
                                     DirectedEdge{max.at(0), add.at(1)},
                                     DirectedEdge{sep.at(2), add.at(1)},
                                     DirectedEdge{avg.at(0), add.at(2)},
                                     DirectedEdge{sep.at(3), add.at(2)},
                                     DirectedEdge{max.at(1), add.at(3)},
                                     DirectedEdge{sep.at(4), add.at(3)},
                                     DirectedEdge{avg.at(1), add.at(4)},
                                     DirectedEdge{id.at(0), add.at(4)},
                                     DirectedEdge{add.at(0), sep.at(4)},
                                     DirectedEdge{add.at(0), avg.at(1)},
                                     DirectedEdge{add.at(1), id.at(0)},
                                     DirectedEdge{add.at(2), concat.at(0)},
                                     DirectedEdge{add.at(3), concat.at(0)},
                                     DirectedEdge{add.at(4), concat.at(0)}};

  add_edges(g, edges);

  assert(get_sinks(g).size() == 1);
  assert(get_sources(g).size() == 2);
  assert(is_acyclic(g));
  return {g, inputs.at(0), inputs.at(1)};
}

DiGraph make_cifar10(size_t num_reduction_cells, size_t N) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  Node input = g.add_node();
  std::deque<Node> outputting = {input, input, input};
  std::deque<Node> inputting;
  size_t num_cells = num_reduction_cells + N * (num_reduction_cells + 1);
  for (int i = 0; i < num_cells; i++) {
    auto [s, earlier_input, later_input] = (i % (N + 1) == N)
                                               ? make_reduction_nasnet_cell()
                                               : make_normal_nasnet_cell();
    Node cell_output = get_only(get_sinks(s));
    std::unordered_map<Node, Node> node_map = parallel_extend(g, s);
    later_input = node_map.at(later_input);
    earlier_input = node_map.at(earlier_input);
    cell_output = node_map.at(cell_output);

    outputting.push_back(cell_output);
    outputting.push_back(cell_output);
    inputting.push_back(earlier_input);
    inputting.push_back(later_input);

    Node a = outputting.front();
    Node b = inputting.front();
    inputting.pop_front();
    outputting.pop_front();
    g.add_edge(DirectedEdge{a, b});

    a = outputting.front();
    b = inputting.front();
    inputting.pop_front();
    outputting.pop_front();
    g.add_edge(DirectedEdge{a, b});

    assert(is_2_terminal_dag(g));
    assert(inputting.size() == 0);
    assert(outputting.size() == 3);
  }
  return g;
}

DiGraph make_linear(size_t length) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  if (length == 0) {
    return g;
  }
  std::vector<Node> nodes = add_nodes(g, length);

  for (size_t i = 0; i < length - 1; ++i) {
    g.add_edge(DirectedEdge{nodes.at(i), nodes.at(i + 1)});
  }

  return g;
}

DiGraph make_rhombus() {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, 4);

  std::vector<DirectedEdge> edges = {DirectedEdge{n.at(0), n.at(1)},
                                     DirectedEdge{n.at(0), n.at(2)},
                                     DirectedEdge{n.at(1), n.at(3)},
                                     DirectedEdge{n.at(2), n.at(3)}};

  add_edges(g, edges);
  return g;
}

DiGraph make_diamond() {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, 6);

  std::vector<DirectedEdge> edges = {
      DirectedEdge{n.at(0), n.at(1)},
      DirectedEdge{n.at(0), n.at(2)},
      DirectedEdge{n.at(1), n.at(3)},
      DirectedEdge{n.at(2), n.at(3)},
      DirectedEdge{n.at(2), n.at(4)},
      DirectedEdge{n.at(3), n.at(5)},
      DirectedEdge{n.at(4), n.at(5)},
  };

  add_edges(g, edges);
  return g;
}

DiGraph make_fully_connected(std::vector<size_t> layer_sizes) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<std::vector<Node>> layers =
      transform(layer_sizes, [&g](size_t size) { return add_nodes(g, size); });

  std::vector<DirectedEdge> edges;

  for (size_t i = 0; i < layers.size() - 1; ++i) {
    for (Node const &n1 : layers.at(i)) {
      for (Node const &n2 : layers.at(i + 1)) {
        edges.push_back(DirectedEdge{n1, n2});
      }
    }
  }

  add_edges(g, edges);
  return g;
}

DiGraph make_parallel_chains(size_t chain_length, size_t chain_num) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  assert(chain_length >= 3);
  assert(chain_num >= 1);
  std::vector<std::vector<Node>> chains;

  for (size_t i = 0; i < chain_num; i++) {
    std::vector<Node> chain_nodes = add_nodes(g, chain_length - 2);
    chains.push_back(chain_nodes);

    for (size_t j = 0; j < chain_length - 3; j++) {
      g.add_edge(DirectedEdge{chain_nodes.at(j), chain_nodes.at(j + 1)});
    }
  }

  Node source = g.add_node();
  Node sink = g.add_node();

  for (std::vector<Node> const &chain : chains) {
    g.add_edge(DirectedEdge{source, chain.front()});
    g.add_edge(DirectedEdge{chain.back(), sink});
  }

  return g;
}

DiGraph make_sample_dag_1() {

  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, 7);
  std::vector<DirectedEdge> edges = {DirectedEdge{n.at(0), n.at(1)},
                                     DirectedEdge{n.at(0), n.at(2)},
                                     DirectedEdge{n.at(2), n.at(3)},
                                     DirectedEdge{n.at(1), n.at(4)},
                                     DirectedEdge{n.at(3), n.at(4)},
                                     DirectedEdge{n.at(3), n.at(5)},
                                     DirectedEdge{n.at(4), n.at(5)},
                                     DirectedEdge{n.at(0), n.at(6)},
                                     DirectedEdge{n.at(2), n.at(6)},
                                     DirectedEdge{n.at(6), n.at(5)}};
  add_edges(g, edges);
  assert(is_2_terminal_dag(g));
  return g;
}

DiGraph make_sample_dag_2() {
  NOT_IMPLEMENTED();
}

DiGraph make_sample_dag_3() {
  // Taken by "A New Algorithm for Mapping DAGs to Series-ParallelSplit Form,
  // Escribano et Al, 2002"
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, 18);

  std::vector<DirectedEdge> edges = {
      DirectedEdge{n.at(0), n.at(1)},   DirectedEdge{n.at(0), n.at(2)},
      DirectedEdge{n.at(1), n.at(3)},   DirectedEdge{n.at(1), n.at(4)},
      DirectedEdge{n.at(2), n.at(10)},  DirectedEdge{n.at(2), n.at(11)},
      DirectedEdge{n.at(2), n.at(12)},  DirectedEdge{n.at(3), n.at(5)},
      DirectedEdge{n.at(3), n.at(6)},   DirectedEdge{n.at(4), n.at(6)},
      DirectedEdge{n.at(4), n.at(7)},   DirectedEdge{n.at(4), n.at(10)},
      DirectedEdge{n.at(5), n.at(8)},   DirectedEdge{n.at(6), n.at(8)},
      DirectedEdge{n.at(6), n.at(9)},   DirectedEdge{n.at(7), n.at(8)},
      DirectedEdge{n.at(8), n.at(17)},  DirectedEdge{n.at(9), n.at(17)},
      DirectedEdge{n.at(10), n.at(16)}, DirectedEdge{n.at(11), n.at(16)},
      DirectedEdge{n.at(12), n.at(13)}, DirectedEdge{n.at(12), n.at(14)},
      DirectedEdge{n.at(13), n.at(15)}, DirectedEdge{n.at(14), n.at(15)},
      DirectedEdge{n.at(15), n.at(16)}, DirectedEdge{n.at(16), n.at(17)}};

  add_edges(g, edges);
  return g;
}

DiGraph make_taso_nasnet_cell() {
  // From the TASO paper, pg 57
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  Node root = g.add_node();
  std::vector<Node> input = add_nodes(g, 2);
  std::vector<Node> dwc = add_nodes(g, 5);
  std::vector<Node> conv = add_nodes(g, 5);
  std::vector<Node> avg = add_nodes(g, 3);
  std::vector<Node> add = add_nodes(g, 5);
  Node concat = g.add_node();

  std::vector<DirectedEdge> edges = {DirectedEdge{root, input.at(0)},
                                     DirectedEdge{root, input.at(1)},
                                     DirectedEdge{input.at(0), dwc.at(0)},
                                     DirectedEdge{input.at(0), dwc.at(1)},
                                     DirectedEdge{input.at(0), avg.at(0)},
                                     DirectedEdge{input.at(0), avg.at(1)},
                                     DirectedEdge{input.at(0), avg.at(2)},
                                     DirectedEdge{input.at(0), dwc.at(2)},
                                     DirectedEdge{input.at(1), add.at(2)},
                                     DirectedEdge{input.at(1), dwc.at(3)},
                                     DirectedEdge{input.at(1), dwc.at(4)},
                                     DirectedEdge{input.at(1), add.at(4)},
                                     DirectedEdge{dwc.at(0), conv.at(0)},
                                     DirectedEdge{dwc.at(1), conv.at(1)},
                                     DirectedEdge{dwc.at(2), conv.at(2)},
                                     DirectedEdge{dwc.at(3), conv.at(3)},
                                     DirectedEdge{dwc.at(4), conv.at(4)},
                                     DirectedEdge{conv.at(0), add.at(0)},
                                     DirectedEdge{conv.at(1), add.at(0)},
                                     DirectedEdge{avg.at(0), add.at(1)},
                                     DirectedEdge{avg.at(1), add.at(1)},
                                     DirectedEdge{avg.at(2), add.at(2)},
                                     DirectedEdge{conv.at(2), add.at(3)},
                                     DirectedEdge{conv.at(3), add.at(3)},
                                     DirectedEdge{conv.at(4), add.at(4)}};

  add_edges(g, edges);

  for (Node const &a : add) {
    g.add_edge(DirectedEdge{a, concat});
  }
  return g;
}

DiGraph make_2_terminal_random_dag(size_t num_nodes, float p, size_t step) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  Bernoulli sampler = Bernoulli(p);
  std::vector<Node> n = add_nodes(g, num_nodes - 2);
  for (int i = 0; i < n.size(); i++) {
    for (int j = i + step + 1; j < n.size(); j++) {
      if (sampler()) {
        g.add_edge(DirectedEdge{n.at(i), n.at(j)});
      }
    }
  }
  std::unordered_set<Node> sinks = get_sinks(g);
  std::unordered_set<Node> sources = get_sources(g);
  Node sink = g.add_node();
  Node source = g.add_node();
  for (Node s : sources) {
    g.add_edge(DirectedEdge{source, s});
  }
  for (Node s : sinks) {
    g.add_edge(DirectedEdge{s, sink});
  }
  assert(is_2_terminal_dag(g));
  return g;
}

} // namespace FlexFlow

#endif
