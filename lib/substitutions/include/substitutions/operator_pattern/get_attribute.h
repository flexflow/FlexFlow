#ifndef _FLEXFLOW_SUBSTITUTIONS_GET_ATTRIBUTES_H
#define _FLEXFLOW_SUBSTITUTIONS_GET_ATTRIBUTES_H

#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_key.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_value.dtg.h"
#include <optional>

namespace FlexFlow {

std::optional<OperatorAttributeValue> get_attribute(PCGOperatorAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(BatchMatmulAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(BatchNormAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(BroadcastAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(CastAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(CombineAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ConcatAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(Conv2DAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ElementBinaryAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ElementUnaryAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(DropoutAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(EmbeddingAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(FlatAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(GatherAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(InputAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(LayerNormAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(LinearAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue>
    get_attribute(MultiHeadAttentionAttrs const &, OperatorAttributeKey);

std::optional<OperatorAttributeValue> get_attribute(NoopAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(Pool2DAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ReduceAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ReductionAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(RepartitionAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ReplicateAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ReshapeAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ReverseAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(SplitAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(SoftmaxAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(TopKAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(TransposeAttrs const &,
                                                    OperatorAttributeKey);
// optional<OperatorAttributeValue> get_attribute(FusedParallelOpAttrs const &,
//                                                OperatorAttributeKey);

} // namespace FlexFlow

#endif
