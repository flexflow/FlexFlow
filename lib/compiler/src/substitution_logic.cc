#include "op-attrs/operator_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "substitutions/get_attribute.h"
#include "substitutions/substitutions_v2.h"

// using namespace ::FlexFlow::opmeta;
using namespace ::FlexFlow::substitutions;

namespace FlexFlow {
namespace ffc {

bool satisfies(PCGOperatorAttrs const &,
               std::vector<ParallelTensorShape> const &,
               OperatorAttributeConstraint const &);

/* tl::optional<bool> satisfies(OperatorParameters const &params,
 * OperatorConstraint const &constraint) { */

/* } */

/* struct SatisfiesAttributeConstraint { */
/*   SatisfiesAttributeConstraint(OperatorAttributeConstraint const &constraint)
 */
/*     : constraint(constraint) */
/*   { } */

/*   tl::optional<bool> satisfies(OperatorParameters const &attrs) */

/* private: */
/*   OperatorAttributeConstraint constraint; */
/* }; */

/* tl::optional< */

/* tl::optional<OperatorAttributeValue> get_attribute( */

/* tl::optional<OperatorATtributeValue> get_attribute(Conv2D */

template <typename T, typename V>
tl::optional<V>
    evaluate_list_index_access(ListIndexAccess<T> const &index_access,
                               tl::optional<V> const &v) {
  if (!v.has_value() ||
      !mpark::holds_alternative<std::vector<int>>(v.value())) {
    return tl::nullopt;
  }

  auto vec = mpark::get<std::vector<int>>(v.value());
  if (index_access.index >= vec.size()) {
    return tl::nullopt;
  }

  return vec.at(index_access.index);
}

template <typename V>
tl::optional<V> evaluate_list_size(tl::optional<V> const &v) {
  if (!v.has_value() ||
      !mpark::holds_alternative<std::vector<int>>(v.value())) {
    return tl::nullopt;
  }

  return (int)mpark::get<std::vector<int>>(v.value()).size();
}

struct EvaluateOperatorAttributeExpr {
  EvaluateOperatorAttributeExpr(PCGOperatorAttrs const &attrs) : attrs(attrs) {}

  tl::optional<OperatorAttributeValue> operator()(OperatorAttributeKey key) {
    return get_attribute(this->attrs, key);
  }

  tl::optional<OperatorAttributeValue>
      operator()(ListIndexAccess<OperatorAttributeKey> const &index_access) {
    tl::optional<OperatorAttributeValue> v =
        get_attribute(this->attrs, index_access.attribute_key);
    return evaluate_list_index_access(index_access, v);
  }

  tl::optional<OperatorAttributeValue>
      operator()(ListSize<OperatorAttributeKey> const &list_size) {
    tl::optional<OperatorAttributeValue> v =
        get_attribute(this->attrs, list_size.attribute_key);
    return evaluate_list_size(v);
  }

private:
  PCGOperatorAttrs attrs;
};

tl::optional<TensorAttributeValue>
    evaluate_tensor_attribute_expr(ParallelTensorShape const &,
                                   AttributeExpr<TensorAttributeKey> const &);

struct EvaluateTensorAttributeExpr {
  EvaluateTensorAttributeExpr(ParallelTensorShape const &tensor_shape)
      : tensor_shape(tensor_shape) {}

  template <typename T>
  tl::optional<TensorAttributeValue> evaluate(T const &t) {
    return this->operator()(t);
  }

  tl::optional<TensorAttributeValue> operator()(TensorAttributeKey key) {
    switch (key) {
      case TensorAttributeKey::DIM_SIZES: {
        std::vector<int> result;
        for (ParallelDim const &dim : this->tensor_shape) {
          result.push_back(dim.size);
        }
        return result;
      }
      case TensorAttributeKey::DIM_DEGREES: {
        std::vector<int> result;
        for (ParallelDim const &dim : this->tensor_shape) {
          result.push_back(dim.degree);
        }
        return result;
      }
      default:
        throw std::runtime_error("Unknown TensorAttributeKey");
    }
  }

  tl::optional<TensorAttributeValue>
      operator()(ListIndexAccess<TensorAttributeKey> const &index_access) {
    auto v = this->evaluate(index_access.attribute_key);
    return evaluate_list_index_access(index_access, v);
  }

  tl::optional<TensorAttributeValue>
      operator()(ListSize<TensorAttributeKey> const &list_size) {
    return evaluate_list_size(this->evaluate(list_size.attribute_key));
  }

private:
  ParallelTensorShape tensor_shape;
};

tl::optional<TensorAttributeValue>
    evaluate_attribute_expr(ParallelTensorShape const &tensor_shape,
                            AttributeExpr<TensorAttributeKey> const &expr) {
  return mpark::visit(EvaluateTensorAttributeExpr(tensor_shape), expr);
}

tl::optional<OperatorAttributeValue>
    evaluate_attribute_expr(PCGOperatorAttrs const &attrs,
                            AttributeExpr<OperatorAttributeKey> const &expr) {
  return mpark::visit(EvaluateOperatorAttributeExpr(attrs), expr);
}

template <typename V>
tl::optional<bool> satisfies(ConstraintType constraint_type,
                             V const &constraint_value,
                             tl::optional<V> const &maybe_attribute_value) {
  /* tl::optional<V> maybe_attr_val = evalute_attribute_expr(attrs,
   * constraint.attribute_expr); */

  if (!maybe_attribute_value.has_value()) {
    return tl::nullopt;
  }
  V attr_val = maybe_attribute_value.value();

  if (attr_val.index() != constraint_value.index()) {
    return tl::nullopt;
  }

  if (constraint_type == ConstraintType::EQUAL) {
    return attr_val == constraint_value;
  } else {
    throw std::runtime_error("Unknown constraint_type");
  }
}

tl::optional<bool> satisfies(ParallelTensorShape const &tensor_shape,
                             TensorAttributeConstraint const &constraint) {
  auto value = evaluate_attribute_expr(tensor_shape, constraint.attribute_expr);
  return satisfies(
      constraint.constraint_type, constraint.attribute_value, value);
}

tl::optional<bool> satisfies(PCGOperatorAttrs const &params,
                             OperatorAttributeConstraint const &constraint) {
  auto value = evaluate_attribute_expr(params, constraint.attribute_expr);
  return satisfies(
      constraint.constraint_type, constraint.attribute_value, value);
}

template <typename Container, typename Function>
tl::optional<bool> optional_all_of(Container const &container,
                                   Function const &func) {
  for (auto const &element : container) {
    tl::optional<bool> condition = func(element);
    if (!condition.has_value()) {
      return tl::nullopt;
    }

    if (!condition.value()) {
      return false;
    }
  }
  return true;
}

tl::optional<bool> satisfies(PCGOperatorAttrs const &params,
                             OperatorPattern const &pattern) {
  return optional_all_of(pattern.attribute_constraints,
                         [&](OperatorAttributeConstraint const &c) {
                           return satisfies(params, c);
                         });
}

tl::optional<bool> satisfies(ParallelTensorShape const &params,
                             ParallelTensorPattern const &pattern) {
  return optional_all_of(
      pattern.attribute_constraints,
      [&](TensorAttributeConstraint const &c) { return satisfies(params, c); });
}

bool assignment_satisfies(
    IMultiDiGraph const &pcg,
    SubstitutionPattern const &pattern,
    DiGraphPatternMatch const &patternMatch,
    std::unordered_map<Node, PCGOperatorAttrs> const &pcgNodeParams,
    std::unordered_map<OpenMultiDiEdge, ParallelTensorShape> const
        &pcgTensorShapes) {
  bool result = true;
  for (auto const &kv : patternMatch.nodeAssignment) {
    auto patternNode = kv.first;
    auto pcgNode = kv.second;
    tl::optional<bool> constraintResult =
        satisfies(pcgNodeParams.at(pcgNode), pattern.at(patternNode));
    result &= constraintResult.value_or(false);
  }

  for (auto const &kv : patternMatch.edgeAssignment) {
    auto patternEdge = kv.first;
    auto pcgEdge = kv.second;
    tl::optional<bool> constraintResult =
        satisfies(pcgTensorShapes.at(pcgEdge), pattern.at(patternEdge));
    result &= constraintResult.value_or(false);
  }

  result &= pattern_matches(*pattern.graph, pcg, patternMatch);

  return result;
}

} // namespace ffc
} // namespace FlexFlow
