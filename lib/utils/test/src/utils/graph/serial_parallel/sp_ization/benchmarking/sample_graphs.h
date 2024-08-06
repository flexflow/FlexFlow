#ifndef FLEXFLOW_GRAPH_GENERATION_H
#define FLEXFLOW_GRAPH_GENERATION_H

#include "distributions.h"
#include "sample_graphs.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
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

  std::vector<DirectedEdge> edges = {
      DirectedEdge(inputs[0], sep[1]), DirectedEdge(inputs[0], id[1]),
      DirectedEdge(inputs[0], avg[1]), DirectedEdge(inputs[0], avg[2]),
      DirectedEdge(inputs[0], sep[3]), DirectedEdge(inputs[0], sep[4]),
      DirectedEdge(inputs[1], sep[0]), DirectedEdge(inputs[1], id[0]),
      DirectedEdge(inputs[1], avg[0]), DirectedEdge(inputs[1], sep[2]),
      DirectedEdge(sep[0], add[0]),    DirectedEdge(id[0], add[0]),
      DirectedEdge(sep[1], add[1]),    DirectedEdge(sep[2], add[1]),
      DirectedEdge(avg[0], add[2]),    DirectedEdge(id[1], add[2]),
      DirectedEdge(avg[1], add[3]),    DirectedEdge(avg[2], add[3]),
      DirectedEdge(sep[3], add[4]),    DirectedEdge(sep[4], add[4])};
  add_edges(g, edges);

  for (Node const &a : add) {
    g.add_edge(DirectedEdge{a, concat[0]});
  }

  assert(get_sinks(g).size() == 1);
  assert(get_sources(g).size() == 2);
  assert(is_acyclic(g));
  return {g, inputs[0], inputs[1]};
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

  std::vector<DirectedEdge> edges = {
      DirectedEdge{inputs[0], sep[0]}, DirectedEdge{inputs[0], sep[2]},
      DirectedEdge{inputs[0], sep[3]}, DirectedEdge{inputs[1], max[1]},
      DirectedEdge{inputs[1], sep[1]}, DirectedEdge{inputs[1], max[0]},
      DirectedEdge{inputs[1], avg[0]}, DirectedEdge{sep[0], add[0]},
      DirectedEdge{sep[1], add[0]},    DirectedEdge{max[0], add[1]},
      DirectedEdge{sep[2], add[1]},    DirectedEdge{avg[0], add[2]},
      DirectedEdge{sep[3], add[2]},    DirectedEdge{max[1], add[3]},
      DirectedEdge{sep[4], add[3]},    DirectedEdge{avg[1], add[4]},
      DirectedEdge{id[0], add[4]},     DirectedEdge{add[0], sep[4]},
      DirectedEdge{add[0], avg[1]},    DirectedEdge{add[1], id[0]},
      DirectedEdge{add[2], concat[0]}, DirectedEdge{add[3], concat[0]},
      DirectedEdge{add[4], concat[0]}};

  add_edges(g, edges);

  assert(get_sinks(g).size() == 1);
  assert(get_sources(g).size() == 2);
  assert(is_acyclic(g));
  return {g, inputs[0], inputs[1]};
}

DiGraph make_cifar10(size_t num_reduction_cells, size_t N) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  Node input = get_only(add_nodes(g, 1));
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
    g.add_edge(DirectedEdge{nodes[i], nodes[i + 1]});
  }

  return g;
}

DiGraph make_rhombus() {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, 4);

  std::vector<DirectedEdge> edges = {DirectedEdge{n[0], n[1]},
                                     DirectedEdge{n[0], n[2]},
                                     DirectedEdge{n[1], n[3]},
                                     DirectedEdge{n[2], n[3]}};

  add_edges(g, edges);
  return g;
}

DiGraph make_diamond() {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, 6);

  std::vector<DirectedEdge> edges = {
      DirectedEdge{n[0], n[1]},
      DirectedEdge{n[0], n[2]},
      DirectedEdge{n[1], n[3]},
      DirectedEdge{n[2], n[3]},
      DirectedEdge{n[2], n[4]},
      DirectedEdge{n[3], n[5]},
      DirectedEdge{n[4], n[5]},
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
    for (Node const &n1 : layers[i]) {
      for (Node const &n2 : layers[i + 1]) {
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
      g.add_edge(DirectedEdge{chain_nodes[j], chain_nodes[j + 1]});
    }
  }

  Node source = get_only(add_nodes(g, 1));
  Node sink = get_only(add_nodes(g, 1));

  for (auto const &chain : chains) {
    g.add_edge(DirectedEdge{source, chain.front()});
    g.add_edge(DirectedEdge{chain.back(), sink});
  }

  return g;
}

DiGraph make_sample_dag_1() {

  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, 7);
  std::vector<DirectedEdge> edges = {DirectedEdge{n[0], n[1]},
                                     DirectedEdge{n[0], n[2]},
                                     DirectedEdge{n[2], n[3]},
                                     DirectedEdge{n[1], n[4]},
                                     DirectedEdge{n[3], n[4]},
                                     DirectedEdge{n[3], n[5]},
                                     DirectedEdge{n[4], n[5]},
                                     DirectedEdge{n[0], n[6]},
                                     DirectedEdge{n[2], n[6]},
                                     DirectedEdge{n[6], n[5]}};
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
      DirectedEdge(n[0], n[1]),   DirectedEdge(n[0], n[2]),
      DirectedEdge(n[1], n[3]),   DirectedEdge(n[1], n[4]),
      DirectedEdge(n[2], n[10]),  DirectedEdge(n[2], n[11]),
      DirectedEdge(n[2], n[12]),  DirectedEdge(n[3], n[5]),
      DirectedEdge(n[3], n[6]),   DirectedEdge(n[4], n[6]),
      DirectedEdge(n[4], n[7]),   DirectedEdge(n[4], n[10]),
      DirectedEdge(n[5], n[8]),   DirectedEdge(n[6], n[8]),
      DirectedEdge(n[6], n[9]),   DirectedEdge(n[7], n[8]),
      DirectedEdge(n[8], n[17]),  DirectedEdge(n[9], n[17]),
      DirectedEdge(n[10], n[16]), DirectedEdge(n[11], n[16]),
      DirectedEdge(n[12], n[13]), DirectedEdge(n[12], n[14]),
      DirectedEdge(n[13], n[15]), DirectedEdge(n[14], n[15]),
      DirectedEdge(n[15], n[16]), DirectedEdge(n[16], n[17])};

  add_edges(g, edges);
  return g;
}

DiGraph make_taso_nasnet_cell() {
  // From the TASO paper, pg 57
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  Node root = get_only(add_nodes(g, 1));
  std::vector<Node> input = add_nodes(g, 2);
  std::vector<Node> dwc = add_nodes(g, 5);
  std::vector<Node> conv = add_nodes(g, 5);
  std::vector<Node> avg = add_nodes(g, 3);
  std::vector<Node> add = add_nodes(g, 5);
  Node concat = get_only(add_nodes(g, 1));

  std::vector<DirectedEdge> edges = {
      DirectedEdge{root, input[0]},   DirectedEdge{root, input[1]},
      DirectedEdge{input[0], dwc[0]}, DirectedEdge{input[0], dwc[1]},
      DirectedEdge{input[0], avg[0]}, DirectedEdge{input[0], avg[1]},
      DirectedEdge{input[0], avg[2]}, DirectedEdge{input[0], dwc[2]},
      DirectedEdge{input[1], add[2]}, DirectedEdge{input[1], dwc[3]},
      DirectedEdge{input[1], dwc[4]}, DirectedEdge{input[1], add[4]},
      DirectedEdge{dwc[0], conv[0]},  DirectedEdge{dwc[1], conv[1]},
      DirectedEdge{dwc[2], conv[2]},  DirectedEdge{dwc[3], conv[3]},
      DirectedEdge{dwc[4], conv[4]},  DirectedEdge{conv[0], add[0]},
      DirectedEdge{conv[1], add[0]},  DirectedEdge{avg[0], add[1]},
      DirectedEdge{avg[1], add[1]},   DirectedEdge{avg[2], add[2]},
      DirectedEdge{conv[2], add[3]},  DirectedEdge{conv[3], add[3]},
      DirectedEdge{conv[4], add[4]}};

  add_edges(g, edges);

  for (auto const &a : add) {
    g.add_edge(DirectedEdge{a, concat});
  }
  return g;
}

DiGraph make_2_terminal_random_dag(size_t num_nodes, float p, size_t step) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  auto sampler = Bernoulli(p);
  std::vector<Node> n = add_nodes(g, num_nodes - 2);
  for (int i = 0; i < n.size(); i++) {
    for (int j = i + step + 1; j < n.size(); j++) {
      if (sampler()) {
        g.add_edge(DirectedEdge{n[i], n[j]});
      }
    }
  }
  auto sinks = get_sinks(g);
  auto sources = get_sources(g);
  auto sink = get_only(add_nodes(g, 1));
  auto source = get_only(add_nodes(g, 1));
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
