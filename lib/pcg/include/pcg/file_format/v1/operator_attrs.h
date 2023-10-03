#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPERATOR_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPERATOR_ATTRS_H

#include "op-attrs/operator_attrs.h"
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
// FIXME: Re-enable this once the JSON issues with std::vector have been
// resolved.
// #include "ops/layer_norm.h"
#include "ops/linear.h"
#include "ops/noop.h"
#include "ops/pool_2d.h"
// FIXME: Re-enable this once the JSON issues with std::vector have been
// resolved.
// #include "ops/reduce.h"
#include "ops/reduction.h"
#include "ops/repartition.h"
#include "ops/replicate.h"
#include "ops/reshape.h"
#include "ops/reverse.h"
#include "ops/softmax.h"
#include "ops/split.h"
#include "ops/topk.h"
// FIXME: Re-enable this once the JSON issues with std::vector have been
// resolved.
// #include "ops/transpose.h"
#include "utils/json.h"
#include "utils/variant.h"

namespace FlexFlow {

// TODO: Re-enable V1LayerNormAttrs, V1ReduceAttrs, V1TransposeAttrs are
// disabled because they use std::vector which triggers an error saying that
// it cannot be serialized to JSON (when it actually ought to be serializable).
// To get this to build for the moment, they have been disabled.
using V1SharedOperatorAttrs = variant<V1AggregateAttrs,
                                      V1AggregateSpecAttrs,
                                      V1BatchMatmulAttrs,
                                      V1BatchNormAttrs,
                                      V1CastAttrs,
                                      V1ConcatAttrs,
                                      V1Conv2DAttrs,
                                      V1DropoutAttrs,
                                      V1ElementBinaryAttrs,
                                      V1ElementScalarUnaryAttrs,
                                      V1ElementUnaryAttrs,
                                      V1EmbeddingAttrs,
                                      V1FlatAttrs,
                                      V1GatherAttrs,
                                      V1Group_byAttrs,
                                      V1InputAttrs,
                                      // V1LayerNormAttrs,
                                      V1LinearAttrs,
                                      V1MultiHeadAttentionAttrs,
                                      V1NoopAttrs,
                                      V1Pool2DAttrs,
                                      // V1ReduceAttrs,
                                      V1ReverseAttrs,
                                      V1ReshapeAttrs,
                                      V1SplitAttrs,
                                      V1SoftmaxAttrs,
                                      V1TopKAttrs
                                      // V1TransposeAttrs
                                      >;

using V1ParallelOperatorAttrs = variant<V1CombineAttrs,
                                        V1ReductionAttrs,
                                        V1RepartitionAttrs,
                                        V1ReplicateAttrs>;

using V1ComputationGraphAttrs =
    variant_join<V1SharedOperatorAttrs, variant<V1BroadcastAttrs>>;
using V1CompGraphOperatorAttrs = V1ComputationGraphAttrs;

using V1PCGOperatorAttrs =
    variant_join<V1SharedOperatorAttrs, V1ParallelOperatorAttrs>;

V1CompGraphOperatorAttrs to_v1(CompGraphOperatorAttrs const &attrs);
V1PCGOperatorAttrs to_v1(PCGOperatorAttrs const &attrs);

} // namespace FlexFlow

#endif
