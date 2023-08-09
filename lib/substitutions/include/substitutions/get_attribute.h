#ifndef _FLEXFLOW_SUBSTITUTIONS_OPERATOR_ATTRIBUTES_H
#define _FLEXFLOW_SUBSTITUTIONS_OPERATOR_ATTRIBUTES_H

#include "op-attrs/operator_attrs.h"
#include "substitutions/substitutions_v2.h"
#include "tl/optional.hpp"

namespace FlexFlow {
namespace substitutions {

tl::optional<OperatorAttributeValue> get_attribute(PCGOperatorAttrs const &,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(AggregateAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(AggregateSpecAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(BatchMatmulAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(CastAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(CombineAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(ConcatAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(Conv2DAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(ElementBinaryAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(ElementUnaryAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(DropoutAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(ElementBinaryAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(ElementUnaryAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(EmbeddingAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(FlatAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(GatherAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(Group_byAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(LayerNormAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(LinearAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue>
    get_attribute(MultiHeadAttentionAttrs const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(Pool2DAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(ReduceAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(ReductionAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(RepartitionAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(ReplicateAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(ReshapeAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(SplitAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(SoftmaxAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(TopKAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(TransposeAttrs const &p,
                                                   OperatorAttributeKey);
tl::optional<OperatorAttributeValue>
    get_attribute(FusedParallelOpAttrs const &p, OperatorAttributeKey);

} // namespace substitutions
} // namespace FlexFlow

#endif
