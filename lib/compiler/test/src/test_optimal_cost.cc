// #include "compiler/unity_algorithm.h"
// #include "doctest/doctest.h"
// #include "test_cost_estimator.h"

// using namespace FlexFlow;

// TEST_SUITE(FF_TEST_SUITE) {
//   // Rapidcheck infrastructures for graphs does not work for now
//   /*
//   Tests whether optimal_cost can give a valid result given random PCG,
//   trivial allowed machine views, trivial cost estimator and random machine
//   specification.
//   */
//   // TEST_CASE("optimal_cost") {
//   //   auto test_allowed_machine_views = [](Operator const &,
//   //                                        MachineSpecification const &) {
//   //     return std::unordered_set<MachineView>{make_1d_machine_view(0, 1,
//   1)};
//   //   };
//   //   RC_SUBCASE([](ParallelComputationGraph const &g,
//   //                MachineSpecification const &machine_spec) {
//   //     OptimalCostCache cached_subgraph_costs;
//   //     OptimalCostResult result = optimal_cost(g,
//   //                                             test_allowed_machine_views,
//   //                                             TestCostEstimator{},
//   //                                             machine_spec,
//   //                                             cached_subgraph_costs);
//   //     RC_ASSERT(result.runtime > 0);
//   //     RC_ASSERT(keys(result.machine_mapping.machine_views) ==
//   get_nodes(g));
//   //   });
//   // }

//   TEST_CASE("optimal_cost_0") {
//     auto pcg =
//         OutputLabelledMultiDiGraph<Operator, ParallelTensor>::template
//         create<
//             UnorderedOutputLabelledMultiDiGraph<Operator, ParallelTensor>>();

//     Node n0 = pcg.add_node(Operator{InputAttrs{}, "input"});
//     Node n1 = pcg.add_node(Operator{
//         LinearAttrs{1, false, DataType::FLOAT, Activation::RELU,
//         std::nullopt}, "linear"});

//     MultiDiEdge e{n1, pcg.add_node_port(), n0, pcg.add_node_port()};
//     pcg.add_edge(e);
//     ParallelDim dim = {2, 1, false};
//     ParallelTensorDims dims = {FFOrdered<ParallelDim>{dim}};
//     pcg.add_output(e, ParallelTensor(dims, DataType::FLOAT,
//     CreateGrad::YES));

//     auto test_allowed_machine_views = [](Operator const &,
//                                          MachineSpecification const &) {
//       return std::unordered_set<MachineView>{
//           make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))};
//     };

//     CostEstimator estimator = CostEstimator::create<TestCostEstimator>();

//     MachineSpecification machine_spec{1, 1, 1, 1, 1};

//     OptimalCostCache cached_results;

//     OptimalCostResult result = optimal_cost(ParallelComputationGraph(pcg),
//                                             test_allowed_machine_views,
//                                             estimator,
//                                             machine_spec,
//                                             cached_results);

//     CHECK(bool(result.runtime > 0));
//   }
// }
