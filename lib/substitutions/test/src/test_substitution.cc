#include "doctest/doctest.h"
#include "op-attrs/get_op_type.h"
#include "rapidcheck.h"
#include "substitutions/substitution.h"

using namespace FlexFlow;

// TEST_SUITE(FF_TEST_SUITE) {
//   TEST_CASE("substitution") {
// PCGPattern pattern;
// OutputGraphExpr output_expr;
// bidict<DataflowGraphInput
// Substitution s;
//   }
// }

// TEST_SUITE(FF_TEST_SUITE) {
//   TEST_CASE("apply_substitution") {
//     OperatorPattern operator_pattern_n0{
//         std::vector<OperatorAttributeConstraint>{
//             OperatorAttributeConstraint{ConstraintType::EQUAL,
//                                         OperatorAttributeKey::OP_TYPE,
//                                         OperatorType::LINEAR}}};
//
//     ParallelTensorPattern tensor_pattern_e0{
//         std::vector<TensorAttributeConstraint>{
//             TensorAttributeConstraint{ConstraintType::EQUAL,
//                                       ListIndexAccess<TensorAttributeKey>{
//                                           TensorAttributeKey::DIM_SIZES, 0},
//                                       2}}};
//
//     ParallelTensorPattern tensor_pattern_empty{
//         std::vector<TensorAttributeConstraint>{}};
//
//     auto ig =
//         OutputLabelledOpenMultiDiGraph<OperatorPattern,
//         ParallelTensorPattern>::
//             create<UnorderedOutputLabelledOpenMultiDiGraph<
//                 OperatorPattern,
//                 ParallelTensorPattern>>();
//     Node n0 = ig.add_node(operator_pattern_n0);
//     NodePort p0 = ig.add_node_port();
//     InputMultiDiEdge e0{n0, p0, std::make_pair(p0.value(), p0.value())};
//     ig.add_edge(e0);
//     ig.add_label(e0, tensor_pattern_e0);
//
//     RC_ASSERT(get_nodes(ig).size() == 1);
//     RC_ASSERT(get_edges(ig).size() == 1);
//
//     GraphPattern input_graph{ig};
//
//     OperatorAttrAssignment op_ass_n1{
//         {{OperatorAttributeKey::OP_TYPE,
//           AttrConstant{OperatorType::REPARTITION}},
//          {OperatorAttributeKey::PARALLEL_DIM, AttrConstant{ff_dim_t{0}}},
//          {OperatorAttributeKey::PARALLEL_DEGREE, AttrConstant{2}}}};
//
//     OperatorAttrAssignment op_ass_n2{
//         {{OperatorAttributeKey::OP_TYPE, AttrConstant{OperatorType::LINEAR}},
//          {OperatorAttributeKey::OUT_CHANNELS,
//           OperatorAttrAccess{n0, OperatorAttributeKey::OUT_CHANNELS}},
//          {OperatorAttributeKey::USE_BIAS,
//           OperatorAttrAccess{n0, OperatorAttributeKey::USE_BIAS}},
//          {OperatorAttributeKey::DATA_TYPE,
//           OperatorAttrAccess{n0, OperatorAttributeKey::DATA_TYPE}},
//          {OperatorAttributeKey::ACTIVATION,
//           OperatorAttrAccess{n0, OperatorAttributeKey::ACTIVATION}},
//          {OperatorAttributeKey::REGULARIZER,
//           OperatorAttrAccess{n0, OperatorAttributeKey::REGULARIZER}}}};
//
//     OperatorAttrAssignment op_ass_n3{
//         {{OperatorAttributeKey::OP_TYPE,
//         AttrConstant{OperatorType::REDUCTION}},
//          {OperatorAttributeKey::PARALLEL_DIM, AttrConstant{ff_dim_t{0}}},
//          {OperatorAttributeKey::PARALLEL_DEGREE, AttrConstant{2}}}};
//
//     auto og = NodeLabelledOpenMultiDiGraph<OperatorAttrAssignment>::create<
//         UnorderedNodeLabelledOpenMultiDiGraph<OperatorAttrAssignment>>();
//     Node n1 = og.add_node(op_ass_n1);
//     Node n2 = og.add_node(op_ass_n2);
//     Node n3 = og.add_node(op_ass_n3);
//     NodePort p1 = og.add_node_port();
//     NodePort p2 = og.add_node_port();
//     NodePort p3 = og.add_node_port();
//     InputMultiDiEdge e1{n1, p1, {p1.value(), p1.value()}};
//     MultiDiEdge e2{n2, p2, n1, p1};
//     MultiDiEdge e3{n3, p3, n2, p2};
//     og.add_edge(e1);
//     og.add_edge(e2);
//     og.add_edge(e3);
//     OutputGraphExpr output_graph_expr{og};
//
//     RC_ASSERT(get_nodes(og).size() == 3);
//     RC_ASSERT(get_edges(og).size() == 3);
//
//     bidict<InputMultiDiEdge, InputMultiDiEdge> input_mapping;
//     input_mapping.equate(e0, e1);
//     bidict<OutputMultiDiEdge, OutputMultiDiEdge> output_mapping;
//
//     Substitution substitution{
//         input_graph, output_graph_expr, input_mapping, output_mapping};
//
//     SubParallelComputationGraph pcg =
//         OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>::create<
//             UnorderedOutputLabelledOpenMultiDiGraph<Operator,
//                                                     ParallelTensor>>();
//
//     Node n4 = pcg.add_node(Operator{InputAttrs{}, "input"});
//     Node n5 = pcg.add_node(Operator{
//         LinearAttrs{1, false, DataType::FLOAT, Activation::RELU,
//         std::nullopt}, "linear"});
//     NodePort p4 = pcg.add_node_port();
//     NodePort p5 = pcg.add_node_port();
//
//     MultiDiEdge e4{n5, p5, n4, p4};
//     pcg.add_edge(e4);
//     ParallelDim dim = {2, 1, false};
//     ParallelTensorDims dims = {FFOrdered<ParallelDim>{dim}};
//     pcg.add_label(e4, ParallelTensor(dims, DataType::FLOAT,
//     CreateGrad::YES));
//
//     MatchAdditionalCriterion criterion{
//         [&](Node const &pattern_node, Node const &graph_node) {
//           return operator_satisfies(pcg.at(graph_node),
//                                     input_graph.value().at(pattern_node));
//         },
//         [&](OpenMultiDiEdge const &pattern_edge,
//             OpenMultiDiEdge const &graph_edge) {
//           return parallel_tensor_satisfies(
//               pcg.at(graph_edge), input_graph.value().at(pattern_edge));
//         }};
//
//     RC_ASSERT(criterion.node_criterion(n0, n5));
//
//     std::vector<MultiDiGraphPatternMatch> matches =
//         find_pattern_matches(input_graph, pcg, criterion);
//
//     RC_ASSERT(matches.size() == 1);
//
//     SubParallelComputationGraph new_pcg =
//         apply_substitution(pcg, substitution, matches[0]);
//
//     RC_ASSERT(get_nodes(new_pcg).size() == 4);
//     RC_ASSERT(get_edges(new_pcg).size() == 3);
//   }
// }
