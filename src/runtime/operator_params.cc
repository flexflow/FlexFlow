#include "flexflow/operator_params.h"
#include "flexflow/ops/attention.h"
#include "flexflow/ops/cast.h"
#include "flexflow/ops/concat.h"
#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/dropout.h"
#include "flexflow/ops/element_binary.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/embedding.h"
#include "flexflow/ops/flat.h"
#include "flexflow/ops/linear.h"
#include "flexflow/ops/pool_2d.h"
#include "flexflow/ops/reshape.h"
#include "flexflow/ops/softmax.h"
#include "flexflow/ops/transpose.h"
#include "flexflow/parallel_ops/combine.h"
#include "flexflow/parallel_ops/fused_parallel_op.h"
#include "flexflow/parallel_ops/partition.h"
#include "flexflow/parallel_ops/reduction.h"
#include "flexflow/parallel_ops/replicate.h"

namespace FlexFlow {

tl::optional<OperatorParameters> get_op_parameters(Op const *op) {
  switch (op->op_type) {
    case OP_LINEAR:
      return ((Linear *)op)->get_params();
    case OP_CONV2D:
      return ((Conv2D *)op)->get_params();
    case OP_EW_ADD:
    case OP_EW_SUB:
    case OP_EW_MUL:
    case OP_EW_DIV:
      return ((ElementBinary *)op)->get_params();
    case OP_EXP:
    case OP_SCALAR_MULTIPLY:
    case OP_SCALAR_ADD:
    case OP_SCALAR_SUB:
    case OP_SCALAR_TRUE_DIV:
    case OP_RELU:
    case OP_SIGMOID:
    case OP_TANH:
    case OP_IDENTITY:
    case OP_GELU:
    case OP_ELU:
      return ((ElementUnary *)op)->get_params();
    case OP_CONCAT:
      return ((Concat *)op)->get_params();
    case OP_POOL2D:
      return ((Pool2D *)op)->get_params();
    case OP_CAST:
      return ((Cast *)op)->get_params();
    case OP_DROPOUT:
      return ((Dropout *)op)->get_params();
    case OP_EMBEDDING:
      return ((Embedding *)op)->get_params();
    case OP_FLAT:
      return ((Flat *)op)->get_params();
    case OP_MULTIHEAD_ATTENTION:
      return ((MultiHeadAttention *)op)->get_params();
    case OP_RESHAPE:
      return ((Reshape *)op)->get_params();
    case OP_SOFTMAX:
      return ((Softmax *)op)->get_params();
    case OP_REPARTITION:
      return ((Repartition *)op)->get_params();
    case OP_REPLICATE:
      return ((Replicate *)op)->get_params();
    case OP_REDUCTION:
      return ((Reduction *)op)->get_params();
    case OP_COMBINE:
      return ((Combine *)op)->get_params();
    case OP_FUSED_PARALLEL:
      return ((FusedParallelOp *)op)->get_params();
    case OP_TRANSPOSE:
      return ((Transpose *)op)->get_params();
    default:
      return tl::nullopt;
  }
}

}; // namespace FlexFlow
