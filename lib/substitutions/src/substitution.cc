#include "substitutions/substitution.h"

namespace FlexFlow {

struct DeriveValidOperatorAttributeExpr {
  template <typename T>
  std::unordered_set<AttributeExpr<OperatorAttributeKey>>
      operator()(T const &t) {
    return derive_valid_operator_attribute_expr(t);
  }

  std::unordered_set<AttributeExpr<OperatorAttributeKey>>
      derive_valid_operator_attribute_expr(OperatorAttributeKey const &key) {
    return {key};
  }

  std::unordered_set<AttributeExpr<OperatorAttributeKey>>
      derive_valid_operator_attribute_expr(
          ListIndexAccess<OperatorAttributeKey> const &access) {
    return {access, access.attribute_key};
  }

  std::unordered_set<AttributeExpr<OperatorAttributeKey>>
      derive_valid_operator_attribute_expr(
          ListSize<OperatorAttributeKey> const &ls) {
    return {ls, ls.attribute_key};
  }
};

std::unordered_set<AttributeExpr<OperatorAttributeKey>>
    get_valid_operator_attribute_exprs(OperatorPattern const &pattern) {
  return set_union(transform(
      pattern.attribute_constraints, [](OperatorAttributeConstraint const &t) {
        return visit(DeriveValidOperatorAttributeExpr{}, t.attribute_expr);
      }));
}

bool is_valid_operator_attribute_expr(
    OperatorPattern const &pattern,
    AttributeExpr<OperatorAttributeKey> const &expr) {
  return contains(get_valid_operator_attribute_exprs(pattern), expr);
}

struct IsValidOperatorAttributeExprFunctor {
  GraphPattern const &graph_pattern;

  template <typename T>
  bool operator()(T const &t) const {
    return is_valid(t);
  }

  bool is_valid(OperatorAttrAccess const &t) const {
    return is_valid_operator_attribute_expr(graph_pattern.value().at(t.node),
                                            t.attr_expr);
  }

  bool is_valid(AttrConstant const &t) const {
    return true;
  }
};

bool is_valid_operator_attribute_expr(GraphPattern const &pattern,
                                      OperatorAttributeExpr const &expr) {
  return visit(IsValidOperatorAttributeExprFunctor{pattern}, expr);
}

bool is_valid_substitution(Substitution const &s) {
  for (Node const &node : get_nodes(s.output_graph_expr.value())) {
    for (OperatorAttributeExpr expr :
         values(s.output_graph_expr.value().at(node).assignments)) {
      if (!is_valid_operator_attribute_expr(s.input_graph, expr)) {
        return false;
      }
    }
  }
  return true;
}

struct EvaluateOperatorAttributeExpr {
  SubParallelComputationGraph const &graph;
  MultiDiGraphPatternMatch const &match;

  template <typename T>
  OperatorAttributeValue operator()(T const &t) {
    return evaluate(t);
  }

  OperatorAttributeValue evaluate(OperatorAttrAccess const &t) {
    Node node_in_pattern = t.node;
    Node node_in_pcg = match.node_assignment.at_l(node_in_pattern);
    return evaluate_attribute_expr(graph.at(node_in_pcg), t.attr_expr).value();
  }

  OperatorAttributeValue evaluate(AttrConstant const &t) {
    return t.value;
  }
};

OperatorAttributeValue
    evaluate_graph_attribute_expr(SubParallelComputationGraph const &g,
                                  MultiDiGraphPatternMatch const &match,
                                  OperatorAttributeExpr const &expr) {
  return visit(EvaluateOperatorAttributeExpr{g, match}, expr);
}

Operator get_operator_attrs(SubParallelComputationGraph const &graph,
                            MultiDiGraphPatternMatch const &match,
                            OperatorAttrAssignment const &assignment) {
  std::unordered_map<OperatorAttributeKey, OperatorAttributeValue> assignments;
  for (auto const &[key, expr] : assignment.assignments) {
    OperatorAttributeValue value =
        evaluate_graph_attribute_expr(graph, match, expr);
    assignments.emplace(key, value);
  }
  assert(contains_key(assignments, OperatorAttributeKey::OP_TYPE));
  assert(std::holds_alternative<OperatorType>(
      assignments.at(OperatorAttributeKey::OP_TYPE)));
  OperatorType op_type =
      std::get<OperatorType>(assignments.at(OperatorAttributeKey::OP_TYPE));
  switch (op_type) {
    case Op::BATCHMATMUL:
      return Operator{
          BatchMatmulAttrs{std::get<int>(assignments.at(
                               OperatorAttributeKey::A_SEQ_LENGTH_DIM)),
                           std::get<int>(assignments.at(
                               OperatorAttributeKey::B_SEQ_LENGTH_DIM))},
          std::nullopt};
    case Op::BATCHNORM:
      return Operator{BatchNormAttrs{std::get<bool>(
                          assignments.at(OperatorAttributeKey::RELU))},
                      std::nullopt};
    case Op::CAST:
      return Operator{CastAttrs{std::get<DataType>(
                          assignments.at(OperatorAttributeKey::DATA_TYPE))},
                      std::nullopt};
    case Op::CONCAT:
      return Operator{
          ConcatAttrs{
              std::get<ff_dim_t>(assignments.at(OperatorAttributeKey::AXIS)),
              std::get<int>(assignments.at(OperatorAttributeKey::NUM_INPUTS))},
          std::nullopt};
    case Op::CONV2D:
      return Operator{
          Conv2DAttrs{
              std::get<int>(assignments.at(OperatorAttributeKey::OUT_CHANNELS)),
              std::get<int>(assignments.at(OperatorAttributeKey::KERNEL_H)),
              std::get<int>(assignments.at(OperatorAttributeKey::KERNEL_W)),
              std::get<int>(assignments.at(OperatorAttributeKey::STRIDE_H)),
              std::get<int>(assignments.at(OperatorAttributeKey::STRIDE_W)),
              std::get<int>(assignments.at(OperatorAttributeKey::PADDING_H)),
              std::get<int>(assignments.at(OperatorAttributeKey::PADDING_W)),
              std::get<int>(assignments.at(OperatorAttributeKey::GROUPS)),
              std::get<Activation>(
                  assignments.at(OperatorAttributeKey::ACTIVATION)),
              std::get<bool>(assignments.at(OperatorAttributeKey::USE_BIAS))},
          std::nullopt};
    case Op::DROPOUT:
      return Operator{DropoutAttrs{std::get<float>(assignments.at(
                                       OperatorAttributeKey::RATE)),
                                   std::get<unsigned long long>(assignments.at(
                                       OperatorAttributeKey::SEED))},
                      std::nullopt};
    case Op::EW_ADD:
    case Op::EW_DIV:
    case Op::EW_EQUAL:
    case Op::EW_GREATER:
    case Op::EW_LESS:
    case Op::EW_MAX:
    case Op::EW_MIN:
    case Op::EW_MUL:
    case Op::EW_SUB:
      return Operator{
          ElementBinaryAttrs{op_type,
                             std::get<DataType>(assignments.at(
                                 OperatorAttributeKey::DATA_TYPE)),
                             std::get<bool>(assignments.at(
                                 OperatorAttributeKey::SHOULD_BROADCAST_LHS)),
                             std::get<bool>(assignments.at(
                                 OperatorAttributeKey::SHOULD_BROADCAST_RHS))},
          std::nullopt};
    case Op::SCALAR_ADD:
    case Op::SCALAR_FLOOR_DIV:
    case Op::SCALAR_MULTIPLY:
    case Op::SCALAR_SUB:
    case Op::SCALAR_TRUE_DIV:
      return Operator{
          ElementScalarUnaryAttrs{
              op_type,
              std::get<float>(assignments.at(OperatorAttributeKey::SCALAR))},
          std::nullopt};
    case Op::EXP:
    case Op::IDENTITY:
    case Op::GELU:
    case Op::RSQRT:
    case Op::POW:
    case Op::SIN:
    case Op::COS:
      return Operator{ElementUnaryAttrs{op_type}, std::nullopt};
    case Op::EMBEDDING:
      return Operator{
          EmbeddingAttrs{
              std::get<int>(assignments.at(OperatorAttributeKey::NUM_ENTRIES)),
              std::get<int>(assignments.at(OperatorAttributeKey::OUT_CHANNELS)),
              std::get<AggregateOp>(assignments.at(OperatorAttributeKey::AGGR)),
              std::get<DataType>(
                  assignments.at(OperatorAttributeKey::OP_TYPE))},
          std::nullopt};
    case Op::FLAT:
      return Operator{FlatAttrs{}, std::nullopt};
    case Op::GATHER:
      return Operator{GatherAttrs{std::get<ff_dim_t>(
                          assignments.at(OperatorAttributeKey::DIM))},
                      std::nullopt};
    case Op::INPUT:
      return Operator{InputAttrs{}, std::nullopt};
    case Op::LAYERNORM:
      return Operator{
          LayerNormAttrs{
              std::get<stack_vector<ff_dim_t, MAX_TENSOR_DIM>>(
                  assignments.at(OperatorAttributeKey::AXES)),
              std::get<bool>(
                  assignments.at(OperatorAttributeKey::ELEMENTWISE_AFFINE)),
              std::get<float>(assignments.at(OperatorAttributeKey::EPSILON))},
          std::nullopt};
    case Op::LINEAR:
      return Operator{
          LinearAttrs{
              std::get<int>(assignments.at(OperatorAttributeKey::OUT_CHANNELS)),
              std::get<bool>(assignments.at(OperatorAttributeKey::USE_BIAS)),
              std::get<DataType>(
                  assignments.at(OperatorAttributeKey::DATA_TYPE)),
              std::get<Activation>(
                  assignments.at(OperatorAttributeKey::ACTIVATION)),
              std::get<std::optional<RegularizerAttrs>>(
                  assignments.at(OperatorAttributeKey::REGULARIZER))},
          std::nullopt};
    case Op::MULTIHEAD_ATTENTION:
      return Operator{
          MultiHeadAttentionAttrs{
              std::get<int>(assignments.at(OperatorAttributeKey::EMBED_DIM)),
              std::get<int>(assignments.at(OperatorAttributeKey::NUM_HEADS)),
              std::get<int>(assignments.at(OperatorAttributeKey::NUM_HEADS)),
              std::get<int>(assignments.at(OperatorAttributeKey::VDIM)),
              std::get<float>(assignments.at(OperatorAttributeKey::DROPOUT)),
              std::get<bool>(assignments.at(OperatorAttributeKey::BIAS)),
              std::get<bool>(assignments.at(OperatorAttributeKey::ADD_BIAS_KV)),
              std::get<bool>(
                  assignments.at(OperatorAttributeKey::ADD_ZERO_ATTN))},
          std::nullopt};
    case Op::NOOP:
      return Operator{NoopAttrs{}, std::nullopt};
    case Op::POOL2D:
      return Operator{
          Pool2DAttrs{
              std::get<int>(assignments.at(OperatorAttributeKey::KERNEL_H)),
              std::get<int>(assignments.at(OperatorAttributeKey::KERNEL_W)),
              std::get<int>(assignments.at(OperatorAttributeKey::STRIDE_H)),
              std::get<int>(assignments.at(OperatorAttributeKey::STRIDE_W)),
              std::get<int>(assignments.at(OperatorAttributeKey::PADDING_H)),
              std::get<int>(assignments.at(OperatorAttributeKey::PADDING_W)),
              std::get<PoolOp>(assignments.at(OperatorAttributeKey::POOL_TYPE)),
              std::get<Activation>(
                  assignments.at(OperatorAttributeKey::ACTIVATION))},
          std::nullopt};
    case Op::REDUCE_ARGMAX:
    case Op::REDUCE_ARGMIN:
    case Op::REDUCE_MAX:
    case Op::REDUCE_MEAN:
    case Op::REDUCE_MIN:
    case Op::REDUCE_PROD:
    case Op::REDUCE_SUM:
      return Operator{
          ReduceAttrs{
              std::get<stack_vector<ff_dim_t, MAX_TENSOR_DIM>>(
                  assignments.at(OperatorAttributeKey::AXES)),
              op_type,
              std::get<bool>(assignments.at(OperatorAttributeKey::KEEP_DIMS))},
          std::nullopt};
    case Op::REVERSE:
      return Operator{ReverseAttrs{std::get<ff_dim_t>(
                          assignments.at(OperatorAttributeKey::AXIS))},
                      std::nullopt};
    case Op::RESHAPE:
      return Operator{ReshapeAttrs{std::get<TensorShape>(
                          assignments.at(OperatorAttributeKey::SHAPE))},
                      std::nullopt};
    case Op::SPLIT:
      return Operator{
          SplitAttrs{
              std::get<stack_vector<int, MAX_NUM_OUTPUTS>>(
                  assignments.at(OperatorAttributeKey::SPLITS)),
              std::get<ff_dim_t>(assignments.at(OperatorAttributeKey::AXIS))},
          std::nullopt};
    case Op::SOFTMAX:
      return Operator{SoftmaxAttrs{std::get<ff_dim_t>(
                          assignments.at(OperatorAttributeKey::DIM))},
                      std::nullopt};
    case Op::TOPK:
      return Operator{
          TopKAttrs{
              std::get<int>(assignments.at(OperatorAttributeKey::K)),
              std::get<bool>(assignments.at(OperatorAttributeKey::SORTED))},
          std::nullopt};
    case Op::TRANSPOSE:
      return Operator{
          TransposeAttrs{std::get<stack_vector<ff_dim_t, MAX_TENSOR_DIM>>(
              assignments.at(OperatorAttributeKey::PERMUTATION))},
          std::nullopt};
    case Op::COMBINE:
      return Operator{CombineAttrs{std::get<ff_dim_t>(assignments.at(
                                       OperatorAttributeKey::PARALLEL_DIM)),
                                   std::get<int>(assignments.at(
                                       OperatorAttributeKey::PARALLEL_DEGREE))},
                      std::nullopt};
    case Op::REDUCTION:
      return Operator{
          ReductionAttrs{std::get<ff_dim_t>(assignments.at(
                             OperatorAttributeKey::PARALLEL_DIM)),
                         std::get<int>(assignments.at(
                             OperatorAttributeKey::PARALLEL_DEGREE))},
          std::nullopt};
    case Op::REPARTITION:
      return Operator{
          RepartitionAttrs{std::get<ff_dim_t>(assignments.at(
                               OperatorAttributeKey::PARALLEL_DIM)),
                           std::get<int>(assignments.at(
                               OperatorAttributeKey::PARALLEL_DEGREE))},
          std::nullopt};
    case Op::REPLICATE:
      return Operator{
          ReplicateAttrs{std::get<ff_dim_t>(assignments.at(
                             OperatorAttributeKey::PARALLEL_DIM)),
                         std::get<int>(assignments.at(
                             OperatorAttributeKey::PARALLEL_DEGREE))},
          std::nullopt};
    default:
      throw mk_runtime_error("Unknown Operator");
  }
}

struct AddMappedEdgeFunctor {
  bidict<Node, Node> const &node_mapping;
  SubParallelComputationGraph &new_pcg;

  template <typename T>
  void operator()(T const &t) {
    return add_mapped_edge(t);
  }

  void add_mapped_edge(InputMultiDiEdge const &e) {
    new_pcg.add_edge(InputMultiDiEdge{
        node_mapping.at_l(e.dst), new_pcg.add_node_port(), e.uid});
  }

  void add_mapped_edge(OutputMultiDiEdge const &e) {
    new_pcg.add_edge(OutputMultiDiEdge{
        node_mapping.at_l(e.src), new_pcg.add_node_port(), e.uid});
  }

  void add_mapped_edge(MultiDiEdge const &e) {
    new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(e.dst),
                                 new_pcg.add_node_port(),
                                 node_mapping.at_l(e.src),
                                 new_pcg.add_node_port()});
  }
};

struct AddNewEdgeFunctor {
  SubParallelComputationGraph const &old_pcg;
  SubParallelComputationGraph &new_pcg;
  MultiDiGraphPatternMatch const &match;
  bidict<Node, Node> node_mapping;

  template <typename TO, typename TN>
  void operator()(TO const &old_edge, TN const &new_edge) {
    return add_new_edge(old_edge, new_edge);
  }

  void add_new_edge(InputMultiDiEdge const &old_edge,
                    InputMultiDiEdge const &new_edge) {
    new_pcg.add_edge(InputMultiDiEdge{node_mapping.at_l(new_edge.dst),
                                      new_pcg.add_node_port(),
                                      old_edge.uid});
  }

  void add_new_edge(MultiDiEdge const &old_edge,
                    InputMultiDiEdge const &new_edge) {
    new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(new_edge.dst),
                                 new_pcg.add_node_port(),
                                 node_mapping.at_l(old_edge.src),
                                 new_pcg.add_node_port()});
  }

  void add_new_edge(OutputMultiDiEdge const &old_edge,
                    OutputMultiDiEdge const &new_edge) {
    new_pcg.add_edge(OutputMultiDiEdge{node_mapping.at_l(new_edge.src),
                                       new_pcg.add_node_port(),
                                       old_edge.uid});
  }

  void add_new_edge(MultiDiEdge const &old_edge,
                    OutputMultiDiEdge const &new_edge) {
    new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(old_edge.dst),
                                 new_pcg.add_node_port(),
                                 node_mapping.at_l(new_edge.src),
                                 new_pcg.add_node_port()});
  }

  void add_new_edge(InputMultiDiEdge const &, OutputMultiDiEdge const &) {
    assert(false);
  }

  void add_new_edge(OpenMultiDiEdge const &, MultiDiEdge const &) {
    assert(false);
  }

  void add_new_edge(OutputMultiDiEdge const &, InputMultiDiEdge const &) {
    assert(false);
  }
};

SubParallelComputationGraph
    apply_substitution(SubParallelComputationGraph const &pcg,
                       Substitution const &substitution,
                       MultiDiGraphPatternMatch const &match) {
  SubParallelComputationGraph new_pcg =
      OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>::template create<
          UnorderedOutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>>();
  bidict<Node, Node> node_mapping; // Refactor it with global nodes
  for (Node const &node : get_nodes(pcg)) {
    if (!contains_r(match.node_assignment, node)) {
      node_mapping.equate(node, new_pcg.add_node(pcg.at(node)));
    }
  }
  for (OpenMultiDiEdge const &edge : get_edges(pcg)) {
    if (!contains_r(match.edge_assignment, edge)) {
      visit(AddMappedEdgeFunctor{node_mapping, new_pcg}, edge);
    }
  }
  for (Node const &output_node :
       get_nodes(substitution.output_graph_expr.value())) {
    Operator new_op = get_operator_attrs(
        pcg, match, substitution.output_graph_expr.value().at(output_node));
    Node new_node = new_pcg.add_node(new_op);
    node_mapping.equate(output_node, new_node);
  }
  for (OpenMultiDiEdge const &output_edge :
       get_edges(substitution.output_graph_expr.value())) {
    if (std::holds_alternative<InputMultiDiEdge>(output_edge)) {
      InputMultiDiEdge e = std::get<InputMultiDiEdge>(output_edge);
      OpenMultiDiEdge original_edge =
          match.edge_assignment.at_l(substitution.input_mapping.at_r(e));
      visit(AddNewEdgeFunctor{pcg, new_pcg, match, node_mapping},
            original_edge,
            output_edge);
    } else if (std::holds_alternative<OutputMultiDiEdge>(output_edge)) {
      OutputMultiDiEdge e = std::get<OutputMultiDiEdge>(output_edge);
      OpenMultiDiEdge original_edge =
          match.edge_assignment.at_l(substitution.output_mapping.at_r(e));
      visit(AddNewEdgeFunctor{pcg, new_pcg, match, node_mapping},
            original_edge,
            output_edge);
    } else {
      assert(std::holds_alternative<MultiDiEdge>(output_edge));
      MultiDiEdge e = std::get<MultiDiEdge>(output_edge);
      new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(e.dst),
                                   new_pcg.add_node_port(),
                                   node_mapping.at_l(e.src),
                                   new_pcg.add_node_port()});
    }
  }

  return new_pcg;
}

} // namespace FlexFlow
