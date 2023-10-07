#ifndef _FLEXFLOW_SUBSTITUTIONS_GET_ATTRIBUTES_H
#define _FLEXFLOW_SUBSTITUTIONS_GET_ATTRIBUTES_H

#include "op-attrs/operator_attrs.h"
#include "operator_pattern.h"
#include "utils/optional.h"

namespace FlexFlow {

optional<OperatorAttributeValue> get_attribute(PCGOperatorAttrs const &,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(AggregateAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(AggregateSpecAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(BatchMatmulAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(CastAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(CombineAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(ConcatAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(Conv2DAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(ElementBinaryAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(ElementUnaryAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(DropoutAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(ElementBinaryAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(ElementUnaryAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(EmbeddingAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(FlatAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(GatherAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(Group_byAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(LayerNormAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(LinearAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(MultiHeadAttentionAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(Pool2DAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(ReduceAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(ReductionAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(RepartitionAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(ReplicateAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(ReshapeAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(SplitAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(SoftmaxAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(TopKAttrs const &p,
                                               OperatorAttributeKey);
optional<OperatorAttributeValue> get_attribute(TransposeAttrs const &p,
                                               OperatorAttributeKey);
// optional<OperatorAttributeValue> get_attribute(FusedParallelOpAttrs const &p,
//                                                OperatorAttributeKey);

} // namespace FlexFlow

#endif
