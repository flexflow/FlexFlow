#ifndef _FLEXFLOW_SUBSTITUTION_LOADER_H
#define _FLEXFLOW_SUBSTITUTION_LOADER_H

#include "op-meta/op-meta.h"
#include "tl/optional.hpp"
#include <fstream>
#include <vector>

namespace FlexFlow {
namespace substitutions {

enum class ParameterAttribute {
  OP_TYPE,            // AnyOp
  NUM_INPUTS,         // AnyOp
  NUM_OUTPUTS,        // AnyOp
  GROUP,              // Conv2D
  KERNEL_H,           // Conv2D, Pool2D
  KERNEL_W,           // Conv2D, Pool2D
  STRIDE_H,           // Conv2D, Pool2D
  STRIDE_W,           // Conv2D, Pool2D
  PADDING_H,          // Conv2D, Pool2D
  PADDING_W,          // Conv2D, Pool2D
  ACTIVATION,         // Conv2D, Pool2D
  NUMDIM,             // Concat, Transpose
  AXIS,               // Concat, Split
  PERM,               // Transpose
  OUTSHUFFLE,         // Transpose
  MERGE_GCONV_COUNT,  // MergeGConv
  AXES,               // Squeeze, Unsqueeze, Reduce*
  KEEP_DIMS,          // Reduce*
  EPSILON,            // BatchNorm
  REPARTITION_DIM,    // Repartition
  REPARTITION_DEGREE, // Repartition
  REPLICATE_DIM,      // Replicate
  REPLICATE_DEGREE,   // Replicate
  COMBINE_DIM,        // Combine
  COMBINE_DEGREE,     // Combine
  REDUCTION_DIM,      // Reduction
  REDUCTION_DEGREE,   // Reduction
  SOFTMAX_DIM,        // Softmax
  NUM_HEADS,          // MultiHeadAttention
  INVALID,
  PARALLEL_DIM,
  PARALLEL_DEGREE,
  PAD,
};

enum class ConstraintType {
  Equal,
  NotEqual,
  LessThan,
  LessThanEqual,
  GreaterThan,
  GreaterThanEqual,
};

struct OperatorAttributeConstraint {
  ParameterAttribute key;
  ConstraintType constraint;
  int value;
};

struct TensorConstraint {};

struct Tensor {
  int opId;
  int tsId;

  std::vector<TensorConstraint> constraints;
};

struct OperatorConstraint {
  OperatorType op_type;
  std::vector<Tensor> inputs;
  std::vector<OperatorConstraint> constraints;

  tl::optional<int> at(ParameterAttribute key) const;
};

struct MapOutput {
  int dstOpId;
  int dstTsId;
  int srcOpId;
  int srcTsId;
};

struct Substitution {
  std::string name;
  std::vector<OperatorConstraint> srcOp;
  std::vector<OperatorConstraint> dstOp;
  std::vector<MapOutput> mappedOutput;
};

struct SubstitutionCollection {
  std::vector<Substitution> substitutions;
};

} // namespace substitutions
} // namespace FlexFlow

#endif
