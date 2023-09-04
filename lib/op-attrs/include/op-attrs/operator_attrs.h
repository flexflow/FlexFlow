#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "op-attrs/ops/core.h"
#include "ops/aggregate.h"
#include "ops/aggregate_spec.h"
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
#include "ops/groupby.h"
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
#include "utils/variant.h"

namespace FlexFlow {

using SharedOperatorAttrs = variant<AggregateAttrs,
                                    AggregateSpecAttrs,
                                    BatchMatmulAttrs,
                                    BatchNormAttrs,
                                    CastAttrs,
                                    ConcatAttrs,
                                    Conv2DAttrs,
                                    DropoutAttrs,
                                    ElementBinaryAttrs,
                                    ElementScalarUnaryAttrs,
                                    ElementUnaryAttrs,
                                    EmbeddingAttrs,
                                    FlatAttrs,
                                    GatherAttrs,
                                    Group_byAttrs,
                                    InputAttrs,
                                    LayerNormAttrs,
                                    LinearAttrs,
                                    MultiHeadAttentionAttrs,
                                    NoopAttrs,
                                    Pool2DAttrs,
                                    ReduceAttrs,
                                    ReverseAttrs,
                                    ReshapeAttrs,
                                    SplitAttrs,
                                    SoftmaxAttrs,
                                    TopKAttrs,
                                    TransposeAttrs>;
CHECK_WELL_BEHAVED_VALUE_TYPE(SharedOperatorAttrs);
CHECK_FMTABLE(SharedOperatorAttrs);

using ParallelOperatorAttrs =
    variant<CombineAttrs, ReductionAttrs, RepartitionAttrs, ReplicateAttrs>;
CHECK_WELL_BEHAVED_VALUE_TYPE(ParallelOperatorAttrs);
CHECK_FMTABLE(ParallelOperatorAttrs);

using ComputationGraphAttrs =
    variant_join<SharedOperatorAttrs, variant<BroadcastAttrs>>;
using CompGraphOperatorAttrs = ComputationGraphAttrs;
CHECK_WELL_BEHAVED_VALUE_TYPE(ComputationGraphAttrs);
CHECK_FMTABLE(ComputationGraphAttrs);

using PCGOperatorAttrs =
    variant_join<SharedOperatorAttrs, ParallelOperatorAttrs>;
CHECK_WELL_BEHAVED_VALUE_TYPE(PCGOperatorAttrs);
CHECK_FMTABLE(PCGOperatorAttrs);

/* OperatorType get_op_type(CompGraphOperatorAttrs const &); */
/* OperatorType get_op_type(PCGOperatorAttrs const &); */

RecordFormatter as_dot(CompGraphOperatorAttrs const &);
RecordFormatter as_dot(PCGOperatorAttrs const &);

std::vector<ParallelTensorShape> get_output_shapes(
    PCGOperatorAttrs const &op_params,
    std::vector<ParallelTensorShape> const &input_tensor_shapes);

bool is_parallel_op(PCGOperatorAttrs const &);
bool is_valid(PCGOperatorAttrs const &,
              std::vector<ParallelTensorShape> const &);

} // namespace FlexFlow

#endif
