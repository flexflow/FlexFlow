#ifndef _FLEXFLOW_TEST_GENERATOR_H
#define _FLEXFLOW_TEST_GENERATOR_H

#include "compiler/machine_mapping.h"
#include "compiler/sub_parallel_computation_graph.h"
#include "rapidcheck.h"

using namespace FlexFlow;

std::unordered_map<Node, Node> parallel_extend(MultiDiGraph &g,
                                               MultiDiGraph const &ext) {
  std::unordered_map<Node, Node> node_map;
  std::unordered_map<NodePort, NodePort> node_port_map;
  for (Node const &node : get_nodes(ext)) {
    node_map[node] = g.add_node();
  }
  for (NodePort const &node_port : get_node_ports(ext)) {
    node_port_map[node_port] = g.add_node_port();
  }
  for (MultiDiEdge const &edge : get_edges(ext)) {
    g.add_edge(MultiDiEdge{node_map[edge.src],
                           node_map[edge.dst],
                           node_map[edge.srcIdx],
                           node_map[edge.dstIdx]});
  }
  return node_map;
}

std::unordered_map<Node, Node> serial_extend(MultiDiGraph &g,
                                             MultiDiGraph const &ext) {
  std::unordered_set<Node> original_sinks = get_sinks(g);
  std::unordered_map<Node, Node> node_map = parallel_extend(g, ext);
  for (Node const &node1 : original_sinks) {
    for (Node const &node2 : get_sources(ext)) {
      g.add_edge(MultiDiEdge{
          node1, node_map[node2], g.add_node_port(), g.add_node_port()});
    }
  }
  return node_map;
}

MultiDiGraph serial_composition(MultiDiGraph const &g1,
                                MultiDiGraph const &g2) {
  MultiDiGraph g = g1;
  serial_extend(g, g2);
  return g;
}

MultiDiGraph parallel_composition(MultiDiGraph const &g1,
                                  MultiDiGraph const &g2) {
  MultiDiGraph g = g1;
  parallel_extend(g, g2);
  return g;
}

struct MultiDiGraphFromSPDecomposition {
  template <typename T>
  MultiDiGraph operator()(T const &t) {
    return multidigraph_from_sp_decomposition(t);
  }
};

MultiDiGraph multidigraph_from_sp_decomposition(
    SerialParallelDecomposition const &sp_decomposition) {
  return visit(MultiDiGraphFromSPDecomposition{}, sp_decomposition);
}

MultiDiGraph multidigraph_from_sp_decomposition(
    variant<Parallel, Node> const &sp_decomposition) {
  return visit(MultiDiGraphFromSPDecomposition{}, sp_decomposition);
}

MultiDiGraph multidigraph_from_sp_decomposition(
    variant<Serial, Node> const &sp_decomposition) {
  return visit(MultiDiGraphFromSPDecomposition{}, sp_decomposition);
}

MultiDiGraph multidigraph_from_sp_decomposition(Serial const &serial) {
  MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
  for (auto child : serial.children) {
    serial_extend(g, multidigraph_from_sp_decomposition(child));
  }
  return g;
}

MultiDiGraph multidigraph_from_sp_decomposition(Parallel const &parallel) {
  MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
  for (auto child : parallel.children) {
    parallel_extend(g, multidigraph_from_sp_decomposition(child));
  }
  return g;
}

MultiDiGraph multidigraph_from_sp_decomposition(Node const &Node) {
  MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
  g.add_node();
  return g;
}

template <typename NodeLabel, typename OutputLabel>
OutputLabelledMultiDiGraph<NodeLabel, OutputLabel>
    generate_test_labelled_sp_graph() {
  NOT_IMPLEMENTED();
  // Is there a way to construct a labelled graph from a MultiDiGraph and the
  // labels?
}

rc::Gen<int> small_integer_generator() {
  return gen::inRange(1, 4);
}

namespace rc {

Gen<MultiDiGraph> serialParallelMultiDiGraph() {
  return gen::map(gen::arbitrary<SerialParallelDecomposition>(),
                  multidigraph_from_sp_decomposition);
}

template <>
struct Arbitrary<variant<Serial, Node>> {
  static Gen<variant<Serial, Node>> arbitrary() {
    return gen::mapcat(gen::arbitrary<bool>(), [](bool is_node) {
      return is_node ? gen::arbitrary<Node>() : gen::arbitrary<Serial>();
    });
  }
};

template <>
struct Arbitrary<variant<Parallel, Node>> {
  static Gen<variant<Parallel, Node>> arbitrary() {
    return gen::mapcat(gen::arbitrary<bool>(), [](bool is_node) {
      return is_node ? gen::arbitrary<Node>() : gen::arbitrary<Parallel>();
    });
  }
};

template <>
struct Arbitrary<Serial> {
  static Gen<Serial> arbitrary() {
    return gen::build<Serial>(
        gen::set(&Serial::children,
                 gen::container<std::vector<variant<Parallel, Node>>>(
                     gen::arbitrary<variant<Parallel, Node>>())));
  }
};

template <>
struct Arbitrary<Parallel> {
  static Gen<Parallel> arbitrary() {
    return gen::build<Parallel>(
        gen::set(&Parallel::children,
                 gen::container<std::vector<variant<Serial, Node>>>(
                     gen::arbitrary<variant<Serial, Node>>())));
  }
};

template <>
struct Arbitrary<SerialParallelDecomposition> {
  static Gen<SerialParallelDecomposition> arbitrary() {
    return gen::mapcat(gen::arbitrary<bool>(), [](bool is_serial) {
      return is_serial ? gen::construct<SerialParallelDecomposition>(
                             gen::arbitrary<Serial>())
                       : gen::construct<SerialParallelDecomposition>(
                             gen::arbitrary<Parallel>());
    });
  }
};

template <typename Tag, typename T>
struct Arbitrary<Tag> {
  static Gen<
      std::enable_if<std::is_base_of<strong_typedef<Tag, T>, Tag>::value>::type>
      arbitrary() {
    return gen::construct<Tag>(gen::arbitrary<T>());
  }
};

template <>
struct Arbitrary<MachineView> {
  static Gen<MachineView> arbitrary() {
    return gen::apply(make_1d_machine_view,
                      gen::arbitrary<gpu_id_t>,
                      gen::arbitrary<gpu_id_t>,
                      small_integer_generator());
  }
}

template <>
struct Arbitrary<MachineMapping> {
  static Gen<MachineMapping> arbitrary() {
    return gen::build<MachineMapping>(
        gen::set(&MachineMapping::runtime, gen::nonZero<float>());
        gen::set(&MachineMapping::machine_views,
                 gen::container<std::unordered_map<Node, MachineView>>(
                     gen::arbitrary<Node>(), gen::arbitrary<MachineView>())));
  }
}

template <>
struct Arbitrary<MachineSpecification> {
  static Gen<MachineSpecification> arbitrary() {
    return gen::build<MachineMapping>(
        gen::set(&MachineSpecification::num_nodes, gen::inRange(1, 64)),
        gen::set(&MachineSpecification::num_cpus_per_node, gen::inRange(1, 64)),
        gen::set(&MachineSpecification::num_gpus_per_node, gen::inRange(1, 16)),
        gen::set(&MachineSpecification::inter_node_bandwidth,
                 gen::nonZero<float>()),
        gen::set(&MachineSpecification::intra_node_bandwidth,
                 gen::nonZero<float>()));
  }
}

} // namespace rc

#endif
