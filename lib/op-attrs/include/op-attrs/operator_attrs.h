#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "ops/attention.h"
#include "ops/batch_matmul.h"
#include "ops/batch_norm.h"
#include "ops/broadcast.h"
#include "ops/cast.h"
#include "ops/combine.h"
#include "ops/concat.h"
#include "ops/conv_2d.h"
#include "ops/dropout.h"
#include "ops/element_binary.h"
#include "ops/element_unary.h"
#include "ops/embedding.h"
#include "ops/flat.h"
#include "ops/gather.h"
#include "ops/input.h"
#include "ops/layer_norm.h"
#include "ops/linear.h"
#include "ops/noop.h"
#include "ops/pool_2d.h"
#include "ops/reduce.h"
#include "ops/reduction.h"
#include "ops/repartition.h"
#include "ops/replicate.h"
#include "ops/reshape.h"
#include "ops/reverse.h"
#include "ops/softmax.h"
#include "ops/split.h"
#include "ops/topk.h"
#include "ops/transpose.h"
#include "utils/record_formatter.h"
#include "utils/variant.h"
#include <variant>

namespace FlexFlow {

std::vector<ParallelTensorShape> get_output_shapes(
    PCGOperatorAttrs const &op_params,
    std::vector<ParallelTensorShape> const &input_tensor_shapes);

bool is_valid(PCGOperatorAttrs const &,
              std::vector<ParallelTensorShape> const &);

} // namespace FlexFlow

#endif
