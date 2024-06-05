#ifndef _FLEXFLOW_SUBSTITUTIONS_GET_ATTRIBUTES_H
#define _FLEXFLOW_SUBSTITUTIONS_GET_ATTRIBUTES_H

#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_key.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_value.dtg.h"
#include <optional>

namespace FlexFlow {

std::optional<OperatorAttributeValue> get_attribute(PCGOperatorAttrs const &,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(BatchMatmulAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(BatchNormAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(CastAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(CombineAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ConcatAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(Conv2DAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ElementBinaryAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ElementUnaryAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(DropoutAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue>
    get_attribute(ElementScalarUnaryAttrs const &p, OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(EmbeddingAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(FlatAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(GatherAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(InputAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(LayerNormAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(LinearAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue>
    get_attribute(MultiHeadAttentionAttrs const &p, OperatorAttributeKey);

std::optional<OperatorAttributeValue> get_attribute(NoopAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(Pool2DAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ReduceAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ReductionAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(RepartitionAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ReplicateAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ReshapeAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(ReverseAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(SplitAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(SoftmaxAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(TopKAttrs const &p,
                                                    OperatorAttributeKey);
std::optional<OperatorAttributeValue> get_attribute(TransposeAttrs const &p,
                                                    OperatorAttributeKey);
// optional<OperatorAttributeValue> get_attribute(FusedParallelOpAttrs const &p,
//                                                OperatorAttributeKey);

} // namespace FlexFlow

#endif
