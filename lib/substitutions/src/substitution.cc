#include "substitutions/substitution.h"

namespace FlexFlow {

/* struct DeriveValidOperatorAttributeExpr { */
/*   template <typename T> */
/*   std::unordered_set<AttributeExpr<OperatorAttributeKey>> */
/*       operator()(T const &t) { */
/*     return derive_valid_operator_attribute_expr(t); */
/*   } */

/*   std::unordered_set<AttributeExpr<OperatorAttributeKey>> */
/*       derive_valid_operator_attribute_expr(OperatorAttributeKey const &key) { */
/*     return {key}; */
/*   } */

/*   std::unordered_set<AttributeExpr<OperatorAttributeKey>> */
/*       derive_valid_operator_attribute_expr( */
/*           ListIndexAccess<OperatorAttributeKey> const &access) { */
/*     return {access, access.attribute_key}; */
/*   } */

/*   std::unordered_set<AttributeExpr<OperatorAttributeKey>> */
/*       derive_valid_operator_attribute_expr( */
/*           ListSize<OperatorAttributeKey> const &ls) { */
/*     return {ls, ls.attribute_key}; */
/*   } */
/* }; */

/* std::unordered_set<AttributeExpr<OperatorAttributeKey>> */
/*     get_valid_operator_attribute_exprs(OperatorPattern const &pattern) { */
/*   return set_union(transform( */
/*       pattern.attribute_constraints, [](OperatorAttributeConstraint const &t) { */
/*         return visit(DeriveValidOperatorAttributeExpr{}, t.attribute_expr); */
/*       })); */
/* } */

/* bool is_valid_operator_attribute_expr( */
/*     OperatorPattern const &pattern, */
/*     AttributeExpr<OperatorAttributeKey> const &expr) { */
/*   return contains(get_valid_operator_attribute_exprs(pattern), expr); */
/* } */

/* struct IsValidOperatorAttributeExprFunctor { */
/*   GraphPattern const &graph_pattern; */

/*   template <typename T> */
/*   bool operator()(T const &t) const { */
/*     return is_valid(t); */
/*   } */

/*   bool is_valid(OperatorAttrAccess const &t) const { */
/*     return is_valid_operator_attribute_expr(graph_pattern.value().at(t.node), */
/*                                             t.attr_expr); */
/*   } */

/*   bool is_valid(AttrConstant const &t) const { */
/*     return true; */
/*   } */
/* }; */

/* bool is_valid_operator_attribute_expr(GraphPattern const &pattern, */
/*                                       OperatorAttributeExpr const &expr) { */
/*   return visit(IsValidOperatorAttributeExprFunctor{pattern}, expr); */
/* } */

/* bool is_valid_substitution(Substitution const &s) { */
/*   for (Node const &node : get_nodes(s.output_graph_expr.value())) { */
/*     for (OperatorAttributeExpr expr : */
/*          values(s.output_graph_expr.value().at(node).assignments)) { */
/*       if (!is_valid_operator_attribute_expr(s.input_graph, expr)) { */
/*         return false; */
/*       } */
/*     } */
/*   } */
/*   return true; */
/* } */

/* struct EvaluateOperatorAttributeExpr { */
/*   SubParallelComputationGraph const &graph; */
/*   MultiDiGraphPatternMatch const &match; */

/*   template <typename T> */
/*   OperatorAttributeValue operator()(T const &t) { */
/*     return evaluate(t); */
/*   } */

/*   OperatorAttributeValue evaluate(OperatorAttrAccess const &t) { */
/*     Node node_in_pattern = t.node; */
/*     Node node_in_pcg = match.node_assignment.at_l(node_in_pattern); */
/*     return evaluate_attribute_expr(graph.at(node_in_pcg), t.attr_expr).value(); */
/*   } */

/*   OperatorAttributeValue evaluate(AttrConstant const &t) { */
/*     return t.value; */
/*   } */
/* }; */

/* OperatorAttributeValue */
/*     evaluate_graph_attribute_expr(SubParallelComputationGraph const &g, */
/*                                   MultiDiGraphPatternMatch const &match, */
/*                                   OperatorAttributeExpr const &expr) { */
/*   return visit(EvaluateOperatorAttributeExpr{g, match}, expr); */
/* } */

/* Operator get_operator_attrs(SubParallelComputationGraph const &graph, */
/*                             MultiDiGraphPatternMatch const &match, */
/*                             OperatorAttrAssignment const &assignment) { */
/*   std::unordered_map<OperatorAttributeKey, OperatorAttributeValue> assignments; */
/*   for (auto const &[key, expr] : assignment.assignments) { */
/*     OperatorAttributeValue value = */
/*         evaluate_graph_attribute_expr(graph, match, expr); */
/*     assignments.emplace(key, value); */
/*   } */
/*   assert(contains_key(assignments, OperatorAttributeKey::OP_TYPE)); */
/*   assert(std::holds_alternative<OperatorType>( */
/*       assignments.at(OperatorAttributeKey::OP_TYPE))); */
/*   OperatorType op_type = */
/*       std::get<OperatorType>(assignments.at(OperatorAttributeKey::OP_TYPE)); */
/*   switch (op_type) { */
/*     case OperatorType::BATCHMATMUL: */
/*       return Operator{ */
/*           BatchMatmulAttrs{std::get<int>(assignments.at( */
/*                                OperatorAttributeKey::A_SEQ_LENGTH_DIM)), */
/*                            std::get<int>(assignments.at( */
/*                                OperatorAttributeKey::B_SEQ_LENGTH_DIM))}, */
/*           std::nullopt}; */
/*     case OperatorType::BATCHNORM: */
/*       return Operator{BatchNormAttrs{std::get<bool>( */
/*                           assignments.at(OperatorAttributeKey::RELU))}, */
/*                       std::nullopt}; */
/*     case OperatorType::CAST: */
/*       return Operator{CastAttrs{std::get<DataType>( */
/*                           assignments.at(OperatorAttributeKey::DATA_TYPE))}, */
/*                       std::nullopt}; */
/*     case OperatorType::CONCAT: */
/*       return Operator{ */
/*           ConcatAttrs{ */
/*               std::get<ff_dim_t>(assignments.at(OperatorAttributeKey::AXIS)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::NUM_INPUTS))}, */
/*           std::nullopt}; */
/*     case OperatorType::CONV2D: */
/*       return Operator{ */
/*           Conv2DAttrs{ */
/*               std::get<int>(assignments.at(OperatorAttributeKey::OUT_CHANNELS)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::KERNEL_H)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::KERNEL_W)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::STRIDE_H)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::STRIDE_W)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::PADDING_H)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::PADDING_W)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::GROUPS)), */
/*               std::get<Activation>( */
/*                   assignments.at(OperatorAttributeKey::ACTIVATION)), */
/*               std::get<bool>(assignments.at(OperatorAttributeKey::USE_BIAS))}, */
/*           std::nullopt}; */
/*     case OperatorType::DROPOUT: */
/*       return Operator{DropoutAttrs{std::get<float>(assignments.at( */
/*                                        OperatorAttributeKey::RATE)), */
/*                                    std::get<unsigned long long>(assignments.at( */
/*                                        OperatorAttributeKey::SEED))}, */
/*                       std::nullopt}; */
/*     case OperatorType::EW_ADD: */
/*     case OperatorType::EW_DIV: */
/*     case OperatorType::EW_EQUAL: */
/*     case OperatorType::EW_GREATER: */
/*     case OperatorType::EW_LESS: */
/*     case OperatorType::EW_MAX: */
/*     case OperatorType::EW_MIN: */
/*     case OperatorType::EW_MUL: */
/*     case OperatorType::EW_SUB: */
/*       return Operator{ */
/*           ElementBinaryAttrs{op_type, */
/*                              std::get<DataType>(assignments.at( */
/*                                  OperatorAttributeKey::DATA_TYPE)), */
/*                              std::get<bool>(assignments.at( */
/*                                  OperatorAttributeKey::SHOULD_BROADCAST_LHS)), */
/*                              std::get<bool>(assignments.at( */
/*                                  OperatorAttributeKey::SHOULD_BROADCAST_RHS))}, */
/*           std::nullopt}; */
/*     case OperatorType::SCALAR_ADD: */
/*     case OperatorType::SCALAR_FLOOR_DIV: */
/*     case OperatorType::SCALAR_MULTIPLY: */
/*     case OperatorType::SCALAR_SUB: */
/*     case OperatorType::SCALAR_TRUE_DIV: */
/*       return Operator{ */
/*           ElementScalarUnaryAttrs{ */
/*               op_type, */
/*               std::get<float>(assignments.at(OperatorAttributeKey::SCALAR))}, */
/*           std::nullopt}; */
/*     case OperatorType::EXP: */
/*     case OperatorType::IDENTITY: */
/*     case OperatorType::GELU: */
/*     case OperatorType::RSQRT: */
/*     case OperatorType::POW: */
/*     case OperatorType::SIN: */
/*     case OperatorType::COS: */
/*       return Operator{ElementUnaryAttrs{op_type}, std::nullopt}; */
/*     case OperatorType::EMBEDDING: */
/*       return Operator{ */
/*           EmbeddingAttrs{ */
/*               std::get<int>(assignments.at(OperatorAttributeKey::NUM_ENTRIES)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::OUT_CHANNELS)), */
/*               std::get<AggregateOp>(assignments.at(OperatorAttributeKey::AGGR)), */
/*               std::get<DataType>( */
/*                   assignments.at(OperatorAttributeKey::OP_TYPE))}, */
/*           std::nullopt}; */
/*     case OperatorType::FLAT: */
/*       return Operator{FlatAttrs{}, std::nullopt}; */
/*     case OperatorType::GATHER: */
/*       return Operator{GatherAttrs{std::get<ff_dim_t>( */
/*                           assignments.at(OperatorAttributeKey::DIM))}, */
/*                       std::nullopt}; */
/*     case OperatorType::INPUT: */
/*       return Operator{InputAttrs{}, std::nullopt}; */
/*     case OperatorType::LAYERNORM: */
/*       return Operator{ */
/*           LayerNormAttrs{ */
/*               std::get<stack_vector<ff_dim_t, MAX_TENSOR_DIM>>( */
/*                   assignments.at(OperatorAttributeKey::AXES)), */
/*               std::get<bool>( */
/*                   assignments.at(OperatorAttributeKey::ELEMENTWISE_AFFINE)), */
/*               std::get<float>(assignments.at(OperatorAttributeKey::EPSILON))}, */
/*           std::nullopt}; */
/*     case OperatorType::LINEAR: */
/*       return Operator{ */
/*           LinearAttrs{ */
/*               std::get<int>(assignments.at(OperatorAttributeKey::OUT_CHANNELS)), */
/*               std::get<bool>(assignments.at(OperatorAttributeKey::USE_BIAS)), */
/*               std::get<DataType>( */
/*                   assignments.at(OperatorAttributeKey::DATA_TYPE)), */
/*               std::get<Activation>( */
/*                   assignments.at(OperatorAttributeKey::ACTIVATION)), */
/*               std::get<std::optional<RegularizerAttrs>>( */
/*                   assignments.at(OperatorAttributeKey::REGULARIZER))}, */
/*           std::nullopt}; */
/*     case OperatorType::MULTIHEAD_ATTENTION: */
/*       return Operator{ */
/*           MultiHeadAttentionAttrs{ */
/*               std::get<int>(assignments.at(OperatorAttributeKey::EMBED_DIM)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::NUM_HEADS)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::NUM_HEADS)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::VDIM)), */
/*               std::get<float>(assignments.at(OperatorAttributeKey::DROPOUT)), */
/*               std::get<bool>(assignments.at(OperatorAttributeKey::BIAS)), */
/*               std::get<bool>(assignments.at(OperatorAttributeKey::ADD_BIAS_KV)), */
/*               std::get<bool>( */
/*                   assignments.at(OperatorAttributeKey::ADD_ZERO_ATTN))}, */
/*           std::nullopt}; */
/*     case OperatorType::NOOP: */
/*       return Operator{NoopAttrs{}, std::nullopt}; */
/*     case OperatorType::POOL2D: */
/*       return Operator{ */
/*           Pool2DAttrs{ */
/*               std::get<int>(assignments.at(OperatorAttributeKey::KERNEL_H)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::KERNEL_W)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::STRIDE_H)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::STRIDE_W)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::PADDING_H)), */
/*               std::get<int>(assignments.at(OperatorAttributeKey::PADDING_W)), */
/*               std::get<PoolOp>(assignments.at(OperatorAttributeKey::POOL_TYPE)), */
/*               std::get<Activation>( */
/*                   assignments.at(OperatorAttributeKey::ACTIVATION))}, */
/*           std::nullopt}; */
/*     case OperatorType::REDUCE_ARGMAX: */
/*     case OperatorType::REDUCE_ARGMIN: */
/*     case OperatorType::REDUCE_MAX: */
/*     case OperatorType::REDUCE_MEAN: */
/*     case OperatorType::REDUCE_MIN: */
/*     case OperatorType::REDUCE_PROD: */
/*     case OperatorType::REDUCE_SUM: */
/*       return Operator{ */
/*           ReduceAttrs{ */
/*               std::get<stack_vector<ff_dim_t, MAX_TENSOR_DIM>>( */
/*                   assignments.at(OperatorAttributeKey::AXES)), */
/*               op_type, */
/*               std::get<bool>(assignments.at(OperatorAttributeKey::KEEP_DIMS))}, */
/*           std::nullopt}; */
/*     case OperatorType::REVERSE: */
/*       return Operator{ReverseAttrs{std::get<ff_dim_t>( */
/*                           assignments.at(OperatorAttributeKey::AXIS))}, */
/*                       std::nullopt}; */
/*     case OperatorType::RESHAPE: */
/*       return Operator{ReshapeAttrs{std::get<TensorShape>( */
/*                           assignments.at(OperatorAttributeKey::SHAPE))}, */
/*                       std::nullopt}; */
/*     case OperatorType::SPLIT: */
/*       return Operator{ */
/*           SplitAttrs{ */
/*               std::get<stack_vector<int, MAX_NUM_OUTPUTS>>( */
/*                   assignments.at(OperatorAttributeKey::SPLITS)), */
/*               std::get<ff_dim_t>(assignments.at(OperatorAttributeKey::AXIS))}, */
/*           std::nullopt}; */
/*     case OperatorType::SOFTMAX: */
/*       return Operator{SoftmaxAttrs{std::get<ff_dim_t>( */
/*                           assignments.at(OperatorAttributeKey::DIM))}, */
/*                       std::nullopt}; */
/*     case OperatorType::TOPK: */
/*       return Operator{ */
/*           TopKAttrs{ */
/*               std::get<int>(assignments.at(OperatorAttributeKey::K)), */
/*               std::get<bool>(assignments.at(OperatorAttributeKey::SORTED))}, */
/*           std::nullopt}; */
/*     case OperatorType::TRANSPOSE: */
/*       return Operator{ */
/*           TransposeAttrs{std::get<stack_vector<ff_dim_t, MAX_TENSOR_DIM>>( */
/*               assignments.at(OperatorAttributeKey::PERMUTATION))}, */
/*           std::nullopt}; */
/*     case OperatorType::COMBINE: */
/*       return Operator{CombineAttrs{std::get<ff_dim_t>(assignments.at( */
/*                                        OperatorAttributeKey::PARALLEL_DIM)), */
/*                                    std::get<int>(assignments.at( */
/*                                        OperatorAttributeKey::PARALLEL_DEGREE))}, */
/*                       std::nullopt}; */
/*     case OperatorType::REDUCTION: */
/*       return Operator{ */
/*           ReductionAttrs{std::get<ff_dim_t>(assignments.at( */
/*                              OperatorAttributeKey::PARALLEL_DIM)), */
/*                          std::get<int>(assignments.at( */
/*                              OperatorAttributeKey::PARALLEL_DEGREE))}, */
/*           std::nullopt}; */
/*     case OperatorType::REPARTITION: */
/*       return Operator{ */
/*           RepartitionAttrs{std::get<ff_dim_t>(assignments.at( */
/*                                OperatorAttributeKey::PARALLEL_DIM)), */
/*                            std::get<int>(assignments.at( */
/*                                OperatorAttributeKey::PARALLEL_DEGREE))}, */
/*           std::nullopt}; */
/*     case OperatorType::REPLICATE: */
/*       return Operator{ */
/*           ReplicateAttrs{std::get<ff_dim_t>(assignments.at( */
/*                              OperatorAttributeKey::PARALLEL_DIM)), */
/*                          std::get<int>(assignments.at( */
/*                              OperatorAttributeKey::PARALLEL_DEGREE))}, */
/*           std::nullopt}; */
/*     default: */
/*       throw mk_runtime_error("Unknown Operator"); */
/*   } */
/* } */

} // namespace FlexFlow
