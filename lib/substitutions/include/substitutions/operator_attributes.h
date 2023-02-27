#ifndef _FLEXFLOW_SUBSTITUTIONS_OPERATOR_ATTRIBUTES_H
#define _FLEXFLOW_SUBSTITUTIONS_OPERATOR_ATTRIBUTES_H

#include "op-meta/operator_params.h"
#include "tl/optional.hpp"
#include "substitutions/substitutions_v2.h"

namespace FlexFlow {
namespace substitutions {

tl::optional<OperatorAttributeValue> get_attribute(opmeta::OperatorParameters const &, OperatorAttributeKey);

tl::optional<OperatorAttributeValue> get_attribute(opmeta::AggregateParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::AggregateSpecParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::BatchMatmulParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::CastParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::CombineParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::ConcatParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::Conv2DParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::ElementBinaryParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::ElementUnaryParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::DropoutParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::ElementBinaryParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::ElementUnaryParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::EmbeddingParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::FlatParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::GatherParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::Group_byParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::LayerNormParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::LinearParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::MultiHeadAttentionParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::Pool2DParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::ReduceParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::ReductionParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::RepartitionParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::ReplicateParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::ReshapeParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::SplitParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::SoftmaxParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::TopKParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::TransposeParams const &p, OperatorAttributeKey);
tl::optional<OperatorAttributeValue> get_attribute(opmeta::FusedParallelOpParams const &p, OperatorAttributeKey);



}
}

#endif 
