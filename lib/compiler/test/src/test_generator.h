#ifndef _FLEXFLOW_TEST_GENERATOR_H
#define _FLEXFLOW_TEST_GENERATOR_H

#include "compiler/machine_mapping.h"
#include "pcg/computation_graph.h"
#include "rapidcheck.h"
#include "substitutions/sub_parallel_computation_graph.h"

using namespace FlexFlow;

// Rapidcheck does not work for now
// /*
//   Generates computation graphs with trivial layers and tensors, which are
//   used for tests focusing on graph structures.
// */
// ComputationGraph test_computataion_graph(MultiDiGraphView const &g) {
//   return materialize_output_labelled_multidigraph_view(
//       ViewMultiDiGraphAsOutputLabelled(
//           g,
//           [](Layer(Node const &)) { return Layer(NoopAttrs{}); },
//           [](Tensor(MultiDiOutput const &)) {
//             return Tensor{0, DataType::FLOAT, nullopt, false, nullopt};
//           }));
// }

// /*
//   Generates parallel computation graphs with trivial layers and tensors,
//   which are used for tests focusing on graph structures.
// */
// ParallelComputationGraph
//     test_parallel_computation_graph(MultiDiGraphView const &g) {
//   return materialize_output_labelled_multidigraph_view(
//       ViewMultiDiGraphAsOutputLabelled(
//           g,
//           [](Operator(Node const &)) { return ParallelTensor(NoopAttrs{}); },
//           [](Operator(MultiDiOutput const &)) {
//             return ParallelTensor(ParallelTensorDims(TensorDims({})),
//                                   DataType::FLOAT);
//           }));
// }

// rc::Gen<int> small_integer_generator() {
//   return rc::gen::inRange(1, 4);
// }

// namespace rc {

// Gen<MultiDiGraph> serialParallelMultiDiGraph() {
//   return gen::map(gen::arbitrary<SeriesParallelDecomposition>(),
//                   multidigraph_from_sp_decomposition);
// }

// template <>
// struct Arbitrary<ComputationGraph> {
//   static Gen<ComputationGraph> arbitrary() {
//     return
//     gen::map(gen::cast<MultiDiGraphView>(serialParallelMultiDiGraph()),
//                     test_computataion_graph);
//   }
// };

// template <>
// struct Arbitrary<ParallelComputationGraph> {
//   static Gen<ParallelComputationGraph> arbitrary() {
//     return
//     gen::map(gen::cast<MultiDiGraphView>(serialParallelMultiDiGraph()),
//                     test_parallel_computation_graph);
//   }
// };

// template <>
// struct Arbitrary<variant<Serial, Node>> {
//   static Gen<variant<Serial, Node>> arbitrary() {
//     return gen::mapcat(gen::arbitrary<bool>(), [](bool is_node) {
//       return is_node
//                  ? gen::cast<variant<Serial, Node>>(gen::arbitrary<Node>())
//                  : gen::cast<variant<Serial,
//                  Node>>(gen::arbitrary<Serial>());
//     });
//   }
// };

// template <>
// struct Arbitrary<variant<Parallel, Node>> {
//   static Gen<variant<Parallel, Node>> arbitrary() {
//     return gen::mapcat(gen::arbitrary<bool>(), [](bool is_node) {
//       return is_node
//                  ? gen::cast<variant<Parallel, Node>>(gen::arbitrary<Node>())
//                  : gen::cast<variant<Parallel, Node>>(
//                        gen::arbitrary<Parallel>());
//     });
//   }
// };

// template <>
// struct Arbitrary<Serial> {
//   static Gen<Serial> arbitrary() {
//     return gen::build<Serial>(
//         gen::set(&Serial::children,
//                  gen::container<std::vector<variant<Parallel, Node>>>(
//                      gen::arbitrary<variant<Parallel, Node>>())));
//   }
// };

// template <>
// struct Arbitrary<Parallel> {
//   static Gen<Parallel> arbitrary() {
//     return gen::build<Parallel>(
//         gen::set(&Parallel::children,
//                  gen::container<std::vector<variant<Serial, Node>>>(
//                      gen::arbitrary<variant<Serial, Node>>())));
//   }
// };

// template <>
// struct Arbitrary<SeriesParallelDecomposition> {
//   static Gen<SeriesParallelDecomposition> arbitrary() {
//     return gen::mapcat(gen::arbitrary<bool>(), [](bool is_serial) {
//       return is_serial ? gen::construct<SeriesParallelDecomposition>(
//                              gen::arbitrary<Serial>())
//                        : gen::construct<SeriesParallelDecomposition>(
//                              gen::arbitrary<Parallel>());
//     });
//   }
// };

// template <typename Tag, typename T>
// struct Arbitrary<Tag> {
//   static Gen<
//       std::enable_if<std::is_base_of<strong_typedef<Tag, T>,
//       Tag>::value>::type> arbitrary() {
//     return gen::construct<Tag>(gen::arbitrary<T>());
//   }
// };

// template <>
// struct Arbitrary<MachineView> {
//   static Gen<MachineView> arbitrary() {
//     return gen::apply(make_1d_machine_view,
//                       gen::arbitrary<gpu_id_t>,
//                       gen::arbitrary<gpu_id_t>,
//                       small_integer_generator());
//   }
// }

// template <>
// struct Arbitrary<MachineMapping> {
//   static Gen<MachineMapping> arbitrary() {
//     return gen::build<MachineMapping>(
//         gen::set(&MachineMapping::machine_views,
//                  gen::container<std::unordered_map<Node, MachineView>>(
//                      gen::arbitrary<Node>(),
//                      gen::arbitrary<MachineView>())));
//   }
// }

// template <>
// struct Arbitrary<MachineSpecification> {
//   static Gen<MachineSpecification> arbitrary() {
//     return gen::build<MachineSpecification>(
//         gen::set(&MachineSpecification::num_nodes, gen::inRange(1, 64)),
//         gen::set(&MachineSpecification::num_cpus_per_node, gen::inRange(1,
//         64)), gen::set(&MachineSpecification::num_gpus_per_node,
//         gen::inRange(1, 16)),
//         gen::set(&MachineSpecification::inter_node_bandwidth,
//                  gen::nonZero<float>()),
//         gen::set(&MachineSpecification::intra_node_bandwidth,
//                  gen::nonZero<float>()));
//   }
// }

// } // namespace rc

#endif
