#ifndef _FLEXFLOW_TEST_GENERATOR_H
#define _FLEXFLOW_TEST_GENERATOR_H

#include "compiler/machine_mapping.h"
#include "compiler/sub_parallel_computation_graph.h"
#include "rapidcheck.h"

using namespace FlexFlow;

enum class CompnType { SERIAL, PARALLEL };

struct Compn {
  CompnType type;
  int component1, component2;
};

int pop_component(std::set<int> const &components, int value) {
  value = value % components.size();
  auto it = components.begin();
  while (value--) {
    it++;
  }
  int component = *it;
  components.erase(it);
  return component;
}

/*
  Generates a series-parallel graph according to the composition sequence
  described by `composition`. A series-parallel graph can be generated as
  follows: 1) Initially, we have E (E is the length of `composition`+1)
  components, each containing a single edge; 2) In iteration `i`, we compose two
  components (`composition[i].component1` and `composition[i].component2`): 2.1)
  If `composition[i].type == SERIAL`, we merge the sink node of component1 and
  the source node of component2; 2.2) If `composition[i].type == PARALLEL`, we
  merge the source nodes and the sink nodes of two components.
*/
MultiDiGraph generate_sp_graph(std::vector<Compn> const &composition) {
  std::set<int> components;
  disjoint_set<Node> node_id; // initially we have 2E nodes, and we will merge
                              // them during the iteration
  std::vector<Node> src,
      dst; // src and dst nodes for each edge before merging
  std::vector<NodePort> srcIdx,
      dstIdx; // src and dst node ports for each edge (I assume it is sufficient
              // to make different edges have different NodePort. Correct me if
              // I am wrong. @lockshaw)
  AdjacencyMultiDiGraph g(0, 0, {});
  for (int i = 0; i <= composition.size(); ++i) {
    components.insert(i);
    src.push_back(g.add_node());
    dst.push_back(g.add_node());
    srcIdx.push_back(g.add_node_port());
    dstIdx.push_back(g.add_node_port());
  }
  std::vector<Node> source_node = src,
                    sink_node =
                        dst; // initially each component has a single edge

  // We compute the src and dst nodes after merging for each edge before
  // actually inserting the edges.

  for (Compn const &compn : composition) {
    int c1 = pop_component(components, compn.component1);
    int c2 = pop_component(components, compn.component2);
    components.insert(c1);
    if (compn.type == CompnType::SERIAL) {
      node_id.m_union(sink_node[c1], source_node[c2]);
      sink_node[c1] = sink_node[c2];
    } else {
      node_id.m_union(source_node[c1], source_node[c2]);
      node_id.m_union(sink_node[c1], sink_node[c2]);
    }
  }

  for (Node node : get_nodes(g)) {
    if (node_id.find(node) != node) {
      g.remove_node_unsafe(node);
    }
  }

  for (int i = 0; i < src.size(); ++i) {
    g.add_edge(MultiDiEdge{src[i], dst[i], srcIdx[i], dstIdx[i]});
  }

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

template <typename Tag, typename T>
struct Arbtrary<Tag> {
  static Gen<
      std::enable_if<std::is_base_of<strong_typedef<Tag, T>, Tag>::value>::type>
      arbitrary() {
    return gen::construct<Tag>(gen::arbitrary<T>());
  }
};

template <>
struct Arbitrary<Node> {
  static Gen<Node> arbitrary() {
    return gen::construct<Node>(gen::arbitrary<size_t>());
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
