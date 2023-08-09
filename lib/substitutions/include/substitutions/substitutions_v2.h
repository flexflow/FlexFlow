#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTIONS_V2_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTIONS_V2_H

#include "graph_pattern.h"
#include "mpark/variant.hpp"
#include "utils/bidict.h"
#include "utils/graph.h"

namespace FlexFlow {
namespace substitutions {

enum class ConstraintType { EQUAL };

enum class OperatorAttributeKey {
  OP_TYPE, // AnyOp
  USE_BIAS,
  GROUPS,
  POOL_TYPE,
  KERNEL_H,
  KERNEL_W,
  DATA_TYPE,
  SCALAR,
  STRIDE_H,
  STRIDE_W,
  PADDING_H,
  PADDING_W,
  AGGR_MODE,
  NUM_ENTRIES,
  OUT_CHANNELS,
  ACTIVATION,
  NUMDIM,
  AXIS,
  PERMUTATION,
  OUTSHUFFLE,
  MERGE_GCONV_COUNT,
  AXES,
  KEEP_DIMS,
  EPSILON,
  PARALLEL_OP_DIM,
  PARALLEL_OP_DEGREE,
  SOFTMAX_DIM,
  NUM_HEADS,
  PARALLEL_DIM,
  PARALLEL_DEGREE,
  PAD,
};

template <typename T>
struct ListIndexAccess {
  T attribute_key;
  int index;
};

template <typename T>
struct ListSize {
  T attribute_key;
};

template <typename T>
using AttributeExpr = mpark::variant<T, ListIndexAccess<T>, ListSize<T>>;

enum class TensorDimensionAttribute { SIZE, DEGREE };

struct TensorNumDimensionsConstraint {
  int value;
};
struct TensorDimensionAttributeConstraint {
  TensorDimensionAttribute attribute;
  int index;
};

enum class TensorAttributeKey { DIM_SIZES, DIM_DEGREES };

using OperatorAttributeValue =
    mpark::variant<int, float, bool, std::vector<int>>;
using TensorAttributeValue = mpark::variant<int, std::vector<int>>;

template <typename K, typename V>
struct AttributeConstraint {
  ConstraintType constraint_type;
  AttributeExpr<K> attribute_expr;
  V attribute_value;
};

using TensorAttributeConstraint =
    AttributeConstraint<TensorAttributeKey, TensorAttributeValue>;
using OperatorAttributeConstraint =
    AttributeConstraint<OperatorAttributeKey, OperatorAttributeValue>;

struct OperatorPattern {
  std::unordered_set<OperatorAttributeConstraint> attribute_constraints;
};

struct ParallelTensorPattern {
  std::unordered_set<TensorAttributeConstraint> attribute_constraints;
};

struct SubstitutionPattern {
  OperatorPattern at(utils::Node) const;
  ParallelTensorPattern at(PatternEdge) const;

  std::unique_ptr<IMultiDiGraphPattern> graph;
  utils::bidict<utils::Node, OperatorPattern> node_map;
  utils::bidict<PatternEdge, ParallelTensorPattern> edge_map;
};

bool assignment_satisfies(
    SubstitutionPattern const &,
    std::unordered_map<utils::Node, utils::Node> const &nodeAssignment,
    std::unordered_map<PatternEdge, utils::MultiDiEdge> const &edgeAssignment);

} // namespace substitutions
} // namespace FlexFlow

#endif
