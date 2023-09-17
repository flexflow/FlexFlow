#include "substitutions/graph_pattern.h"
#include "op-attrs/operator_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph.h"
#include "substitutions/get_attribute.h"
#include "substitutions/graph_pattern_match.h"
#include "substitutions/operator_pattern.h"
#include "substitutions/parallel_tensor_pattern.h"

namespace FlexFlow {

optional<OperatorAttributeValue>
    evaluate_list_index_access(int index,
                               optional<OperatorAttributeValue> const &v) {
  if (!v.has_value() ||
      !holds_alternative<stack_vector<int, MAX_TENSOR_DIM>>(v.value()) ||
      !holds_alternative<stack_vector<ff_dim_t, MAX_TENSOR_DIM>>(v.value())) {
    return nullopt;
  }

  if (index >= MAX_TENSOR_DIM) {
    return nullopt;
  }

  if (holds_alternative<stack_vector<int, MAX_TENSOR_DIM>>(v.value())) {
    return get<stack_vector<int, MAX_TENSOR_DIM>>(v.value()).at(index);
  } else {
    return get<stack_vector<ff_dim_t, MAX_TENSOR_DIM>>(v.value()).at(index);
  }
}

optional<TensorAttributeValue>
    evaluate_list_index_access(int const &index,
                               optional<TensorAttributeValue> const &v) {
  if (!v.has_value() || !holds_alternative<std::vector<int>>(v.value())) {
    return nullopt;
  }

  auto vec = get<std::vector<int>>(v.value());

  if (index >= vec.size()) {
    return nullopt;
  }

  return vec.at(index);
}

optional<OperatorAttributeValue>
    evaluate_list_size(optional<OperatorAttributeValue> const &v) {
  return MAX_TENSOR_DIM;
}

optional<TensorAttributeValue>
    evaluate_list_size(optional<TensorAttributeValue> const &v) {
  if (!v.has_value() || !holds_alternative<std::vector<int>>(v.value())) {
    return nullopt;
  }

  return (int)get<std::vector<int>>(v.value()).size();
}

struct EvaluateOperatorAttributeExpr {
  EvaluateOperatorAttributeExpr(Operator const &attrs) : attrs(attrs) {}

  optional<OperatorAttributeValue> operator()(OperatorAttributeKey const &key) {
    return get_attribute(this->attrs, key);
  }

  optional<OperatorAttributeValue>
      operator()(ListIndexAccess<OperatorAttributeKey> const &index_access) {
    optional<OperatorAttributeValue> v =
        get_attribute(this->attrs, index_access.attribute_key);
    return evaluate_list_index_access(index_access.index, v);
  }

  optional<OperatorAttributeValue>
      operator()(ListSize<OperatorAttributeKey> const &list_size) {
    optional<OperatorAttributeValue> v =
        get_attribute(this->attrs, list_size.attribute_key);
    return evaluate_list_size(v);
  }

private:
  Operator attrs;
};

optional<TensorAttributeValue>
    evaluate_tensor_attribute_expr(ParallelTensor const &,
                                   AttributeExpr<TensorAttributeKey> const &);

struct EvaluateTensorAttributeExpr {
  EvaluateTensorAttributeExpr(ParallelTensor const &tensor_shape)
      : tensor_shape(tensor_shape) {}

  template <typename T>
  optional<TensorAttributeValue> evaluate(T const &t) {
    return this->operator()(t);
  }

  optional<TensorAttributeValue> operator()(TensorAttributeKey key) {
    switch (key) {
      case TensorAttributeKey::DIM_SIZES: {
        std::vector<int> result;
        for (ParallelDim const &dim : this->tensor_shape.dims) {
          result.push_back(dim.size);
        }
        return result;
      }
      case TensorAttributeKey::DIM_DEGREES: {
        std::vector<int> result;
        for (ParallelDim const &dim : this->tensor_shape.dims) {
          result.push_back(dim.degree);
        }
        return result;
      }
      default:
        throw std::runtime_error("Unknown TensorAttributeKey");
    }
  }

  optional<TensorAttributeValue>
      operator()(ListIndexAccess<TensorAttributeKey> const &index_access) {
    optional<TensorAttributeValue> v =
        this->evaluate(index_access.attribute_key);
    return evaluate_list_index_access(index_access.index, v);
  }

  optional<TensorAttributeValue>
      operator()(ListSize<TensorAttributeKey> const &list_size) {
    return evaluate_list_size(this->evaluate(list_size.attribute_key));
  }

private:
  ParallelTensor tensor_shape;
};

optional<TensorAttributeValue>
    evaluate_attribute_expr(ParallelTensor const &tensor_shape,
                            AttributeExpr<TensorAttributeKey> const &expr) {
  return visit(EvaluateTensorAttributeExpr(tensor_shape), expr);
}

optional<OperatorAttributeValue>
    evaluate_attribute_expr(Operator const &attrs,
                            AttributeExpr<OperatorAttributeKey> const &expr) {
  return visit(EvaluateOperatorAttributeExpr(attrs), expr);
}

template <typename V>
optional<bool> satisfies(ConstraintType constraint_type,
                         V const &constraint_value,
                         optional<V> const &maybe_attribute_value) {
  if (!maybe_attribute_value.has_value()) {
    return nullopt;
  }
  V attr_val = maybe_attribute_value.value();

  if (attr_val.index() != constraint_value.index()) {
    return nullopt;
  }

  if (constraint_type == ConstraintType::EQUAL) {
    return attr_val == constraint_value;
  } else {
    throw std::runtime_error("Unknown constraint_type");
  }
}

optional<bool> satisfies(ParallelTensor const &tensor_shape,
                         TensorAttributeConstraint const &constraint) {
  auto value = evaluate_attribute_expr(tensor_shape, constraint.attribute_expr);
  return satisfies(
      constraint.constraint_type, constraint.attribute_value, value);
}

optional<bool> satisfies(Operator const &params,
                         OperatorAttributeConstraint const &constraint) {
  auto value = evaluate_attribute_expr(params, constraint.attribute_expr);
  return satisfies(
      constraint.constraint_type, constraint.attribute_value, value);
}

template <typename Container, typename Function>
optional<bool> optional_all_of(Container const &container,
                               Function const &func) {
  for (auto const &element : container) {
    optional<bool> condition = func(element);
    if (!condition.has_value()) {
      return nullopt;
    }

    if (!condition.value()) {
      return false;
    }
  }
  return true;
}

optional<bool> satisfies(Operator const &params,
                         OperatorPattern const &pattern) {
  return optional_all_of(pattern.attribute_constraints,
                         [&](OperatorAttributeConstraint const &c) {
                           return satisfies(params, c);
                         });
}

optional<bool> satisfies(ParallelTensor const &params,
                         ParallelTensorPattern const &pattern) {
  return optional_all_of(
      pattern.attribute_constraints,
      [&](TensorAttributeConstraint const &c) { return satisfies(params, c); });
}

struct AlwaysTrueCriterion {
  template <typename T>
  bool operator()(T const &t) const {
    return true;
  }
};

bool assignment_satisfies(SubParallelComputationGraph const &pcg,
                          GraphPattern const &pattern,
                          MultiDiGraphPatternMatch const &patternMatch) {
  bool result = true;
  for (auto const &kv : patternMatch.node_assignment) {
    auto patternNode = kv.first;
    auto pcgNode = kv.second;
    optional<bool> constraintResult =
        satisfies(pcg.at(pcgNode), pattern.value().at(patternNode));
    result &= constraintResult.value_or(false);
  }

  for (auto const &kv : patternMatch.edge_assignment) {
    auto patternEdge = kv.first;
    auto pcgEdge = kv.second;
    optional<bool> constraintResult =
        satisfies(at(pcg, pcgEdge), pattern.value().at(patternEdge));
    result &= constraintResult.value_or(false);
  }

  result &= pattern_matches(pattern, pcg, patternMatch, AlwaysTrueCriterion{});

  return result;
}
} // namespace FlexFlow
