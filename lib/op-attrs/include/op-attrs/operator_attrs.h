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

CHECK_VALID_OP_ATTR(AggregateAttrs);

static_assert(is_valid_opattr<AggregateAttrs>::value, "");
static_assert(is_valid_opattr<AggregateSpecAttrs>::value, "");
static_assert(is_valid_opattr<BatchMatmulAttrs>::value, "");
static_assert(is_valid_opattr<CastAttrs>::value, "");
static_assert(is_valid_opattr<ConcatAttrs>::value, "");
static_assert(is_valid_opattr<Conv2DAttrs>::value, "");
static_assert(is_valid_opattr<DropoutAttrs>::value, "");
static_assert(is_valid_opattr<ElementBinaryAttrs>::value, "");
static_assert(is_valid_opattr<ElementScalarUnaryAttrs>::value, "");
static_assert(is_valid_opattr<ElementUnaryAttrs>::value, "");
static_assert(is_valid_opattr<EmbeddingAttrs>::value, "");
static_assert(is_valid_opattr<FlatAttrs>::value, "");
static_assert(is_valid_opattr<GatherAttrs>::value, "");
static_assert(is_valid_opattr<Group_byAttrs>::value, "");
static_assert(is_valid_opattr<InputAttrs>::value, "");
static_assert(is_valid_opattr<LayerNormAttrs>::value, "");
static_assert(is_valid_opattr<LinearAttrs>::value, "");
static_assert(is_valid_opattr<MultiHeadAttentionAttrs>::value, "");
static_assert(is_valid_opattr<NoopAttrs>::value, "");
static_assert(is_valid_opattr<Pool2DAttrs>::value, "");
static_assert(is_valid_opattr<ReduceAttrs>::value, "");
static_assert(is_valid_opattr<ReshapeAttrs>::value, "");
static_assert(is_valid_opattr<SplitAttrs>::value, "");
static_assert(is_valid_opattr<SoftmaxAttrs>::value, "");
static_assert(is_valid_opattr<TopKAttrs>::value, "");
static_assert(is_valid_opattr<TransposeAttrs>::value, "");

using ParallelOperatorAttrs =
    variant<CombineAttrs, ReductionAttrs, RepartitionAttrs, ReplicateAttrs>;

using ComputationGraphAttrs =
    variant_join<SharedOperatorAttrs, variant<BroadcastAttrs>>;
using CompGraphOperatorAttrs = ComputationGraphAttrs;

using PCGOperatorAttrs =
    variant_join<SharedOperatorAttrs, ParallelOperatorAttrs>;

static_assert(is_equal_comparable<ComputationGraphAttrs>::value,
              "ComputationGraphAttrs must support ==");
static_assert(elements_satisfy<is_valid_opattr, ComputationGraphAttrs>::value,
              "");
static_assert(is_neq_comparable<ComputationGraphAttrs>::value,
              "ComputationGraphAttrs must support !=");
static_assert(is_lt_comparable<ComputationGraphAttrs>::value,
              "ComputationGraphAttrs must support <");
static_assert(is_hashable<ComputationGraphAttrs>::value,
              "ComputationGraphAttrs must be hashable");

static_assert(is_equal_comparable<PCGOperatorAttrs>::value,
              "PCGOperatorAttrs must support ==");
static_assert(is_neq_comparable<PCGOperatorAttrs>::value,
              "PCGOperatorAttrs must support !=");
static_assert(is_lt_comparable<PCGOperatorAttrs>::value,
              "PCGOperatorAttrs must support <");
static_assert(is_hashable<PCGOperatorAttrs>::value,
              "PCGOperatorAttrs must be hashable");

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
