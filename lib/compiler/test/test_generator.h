#ifndef _FLEXFLOW_TEST_GENERATOR_H
#define _FLEXFLOW_TEST_GENERATOR_H

#include "compiler/machine_mapping.h"
#include "compiler/sub_parallel_computation_graph.h"
#include "rapidcheck.h"

namespace rc {

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
                      gen::construct<gpu_id_t>(gen::nonZero<int>()),
                      gen::construct<gpu_id_t>(gen::nonZero<int>()),
                      gen::inRange(1, 4));
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
struct Arbitrary<ComputationGraph> {
  static Gen<ComputationGraph> arbitrary() {
    return gen::apply(
        [](int num_nodes,
           std::vector<int> const &lhs,
           std::vector<int> const &rhs,
           std::vector<int> const &compn_type) {
          auto g =
              ::create<UnorderedOutputLabelledMultiDiGraph<Layer, Tensor>>();

          std::vector<Node> nodes, source, sink;

          for (int i = 0; i < num_nodes; ++i) {
            Node new_node = g.add_node(Layer(NoopAttrs{}, nullopt));
            nodes.push_back(new_node);
            source.push_back(new_node);
            sink.push_back(new_node);
          }

          for (int i = 0; i < lhs.size(); ++i) {
            if (i >= rhs.size()) {
              break;
            }
            if (i >= compn_type.size()) {
              break;
            }

            int n0 = lhs[i] % num_nodes, n1 = rhs[i] % num_nodes,
                t = compn_type[i] % 2;

            Node source0 = source[n0], source1 = source[n1];
            Node sink0 = sink[n0], sink1 = sink[n1];

            if (source0 == source1 && sink0 == sink1) {
              continue;
            }

            RC_ASSERT(source0 != source1 && sink0 != sink1);

            if (source0 == sink0 || t == 0) {
              // sequential composition
              g.add_edge(MultiDiOutput{sink0, NodePort(0)},
                         MultiDiInput{source1, NodePort(0)});
              for (int j = 0; j < nodes.size(); ++j) {
                if (source[j] == source1) {
                  source[j] = source0;
                }
                if (sink[j] == sink0) {
                  sink[j] = sink1;
                }
              }
            } else {
              // parallel composition
              g.add_edge(MultiDiOutput{source0, NodePort(0)},
                         MultiDiInput{source1, NodePort(0)});
              g.add_edge(MultiDiOutput{sink1, NodePort(0)},
                         MultiDiInput{sink0, NodePort(0)});
              for (int j = 0; j < nodes.size(); ++j) {
                if (source[j] == source1) {
                  source[j] = source0;
                }
                if (sink[j] == sink1) {
                  sink[j] = sink0;
                }
              }
            }
          }

          return ComputationGraph(g);
        },
        gen::inRange(1, 200),
        gen::arbitrary<vector<int>>(),
        gen::arbitrary<vector<int>>(),
        gen::arbitrary<vector<int>>());
  }
}

template <>
struct Arbitrary<ParallelComputationGraph> {
  static Gen<ParallelComputationGraph> arbitrary() {
    return gen::apply(
        [](int num_nodes,
           std::vector<int> const &lhs,
           std::vector<int> const &rhs,
           std::vector<int> const &compn_type) {
          auto g = OutputLabelledMultiDiGraph::create<
              UnorderedOutputLabelledMultiDiGraph<Operator, ParallelTensor>>();

          std::vector<Node> nodes, source, sink;

          for (int i = 0; i < num_nodes; ++i) {
            Node new_node = g.add_node(Operator(NoopAttrs{}, nullopt));
            nodes.push_back(new_node);
            source.push_back(new_node);
            sink.push_back(new_node);
          }

          for (int i = 0; i < lhs.size(); ++i) {
            if (i >= rhs.size()) {
              break;
            }
            if (i >= compn_type.size()) {
              break;
            }

            int n0 = lhs[i] % num_nodes, n1 = rhs[i] % num_nodes,
                t = compn_type[i] % 2;

            Node source0 = source[n0], source1 = source[n1];
            Node sink0 = sink[n0], sink1 = sink[n1];

            if (source0 == source1 && sink0 == sink1) {
              continue;
            }

            RC_ASSERT(source0 != source1 && sink0 != sink1);

            if (source0 == sink0 || t == 0) {
              // sequential composition
              g.add_edge(MultiDiOutput{sink0, NodePort(0)},
                         MultiDiInput{source1, NodePort(0)});
              for (int j = 0; j < nodes.size(); ++j) {
                if (source[j] == source1) {
                  source[j] = source0;
                }
                if (sink[j] == sink0) {
                  sink[j] = sink1;
                }
              }
            } else {
              // parallel composition
              g.add_edge(MultiDiOutput{source0, NodePort(0)},
                         MultiDiInput{source1, NodePort(0)});
              g.add_edge(MultiDiOutput{sink1, NodePort(0)},
                         MultiDiInput{sink0, NodePort(0)});
              for (int j = 0; j < nodes.size(); ++j) {
                if (source[j] == source1) {
                  source[j] = source0;
                }
                if (sink[j] == sink1) {
                  sink[j] = sink0;
                }
              }
            }
          }

          return ParallelComputationGraph(g);
        },
        gen::inRange(1, 200),
        gen::arbitrary<vector<int>>(),
        gen::arbitrary<vector<int>>(),
        gen::arbitrary<vector<int>>());
  }
}

} // namespace rc

#endif
