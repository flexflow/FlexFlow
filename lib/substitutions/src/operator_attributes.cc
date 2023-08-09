#include "substitutions/operator_attributes.h"
#include "substitutions/substitutions_v2.h"

namespace FlexFlow {
namespace substitutions {

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::AggregateParams const &p, OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::AggregateSpecParams const &p,
                  OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::BatchMatmulParams const &p,
                  OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue> get_attribute(opmeta::CastParams const &p,
                                                   OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::DATA_TYPE:
      return p.dtype;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::CombineParams const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return p.combine_legion_dim;
    case OperatorAttributeKey::PARALLEL_DIM:
      return p.combine_degree;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::ConcatParams const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::AXIS:
      return p.axis;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::Conv2DParams const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::KERNEL_H:
      return p.kernel_h;
    case OperatorAttributeKey::KERNEL_W:
      return p.kernel_w;
    case OperatorAttributeKey::STRIDE_H:
      return p.stride_h;
    case OperatorAttributeKey::STRIDE_W:
      return p.stride_w;
    case OperatorAttributeKey::PADDING_H:
      return p.padding_h;
    case OperatorAttributeKey::PADDING_W:
      return p.padding_w;
    case OperatorAttributeKey::GROUPS:
      return p.groups;
    case OperatorAttributeKey::ACTIVATION:
      return p.activation;
    case OperatorAttributeKey::USE_BIAS:
      return p.use_bias;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::ElementBinaryParams const &p,
                  OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::ElementUnaryParams const &p,
                  OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::SCALAR:
      return p.scalar;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::DropoutParams const &p, OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::EmbeddingParams const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::DATA_TYPE:
      return p.data_type;
    case OperatorAttributeKey::AGGR_MODE:
      return p.aggr;
    case OperatorAttributeKey::NUM_ENTRIES:
      return p.num_entries;
    case OperatorAttributeKey::OUT_CHANNELS:
      return p.out_channels;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue> get_attribute(opmeta::FlatParams const &p,
                                                   OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::GatherParams const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::AXIS:
      return p.legion_dim;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::Group_byParams const &p, OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::LayerNormParams const &p, OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::LinearParams const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OUT_CHANNELS:
      return p.out_channels;
    case OperatorAttributeKey::USE_BIAS:
      return p.use_bias;
    case OperatorAttributeKey::DATA_TYPE:
      return p.data_type;
    case OperatorAttributeKey::ACTIVATION:
      return p.activation;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::MultiHeadAttentionParams const &p,
                  OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::NUM_HEADS:
      return p.num_heads;
    case OperatorAttributeKey::USE_BIAS:
      return p.bias;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::Pool2DParams const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::KERNEL_H:
      return p.kernel_h;
    case OperatorAttributeKey::KERNEL_W:
      return p.kernel_w;
    case OperatorAttributeKey::STRIDE_H:
      return p.stride_h;
    case OperatorAttributeKey::STRIDE_W:
      return p.stride_w;
    case OperatorAttributeKey::PADDING_H:
      return p.padding_h;
    case OperatorAttributeKey::PADDING_W:
      return p.padding_w;
    case OperatorAttributeKey::POOL_TYPE:
      return p.pool_type;
    case OperatorAttributeKey::ACTIVATION:
      return p.activation;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::ReduceParams const &p, OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::ReductionParams const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return p.reduction_legion_dim;
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return p.reduction_degree;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::RepartitionParams const &p,
                  OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return p.repartition_legion_dim;
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return p.repartition_degree;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::ReplicateParams const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return p.replicate_legion_dim;
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return p.replicate_degree;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::ReshapeParams const &p, OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue> get_attribute(opmeta::SplitParams const &p,
                                                   OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::AXIS:
      return p.legion_axis;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::SoftmaxParams const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::AXIS:
      return p.dim;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue> get_attribute(opmeta::TopKParams const &p,
                                                   OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::TransposeParams const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::PERMUTATION:
      return p.perm;
    default:
      return tl::nullopt;
  }
}

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::FusedParallelOpParams const &p,
                  OperatorAttributeKey key) {
  switch (key) {
    default:
      return tl::nullopt;
  }
}

struct GetAttribute {
  GetAttribute(OperatorAttributeKey key) : key(key) {}

  template <typename T>
  tl::optional<OperatorAttributeValue> operator()(T const &t) {
    return get_attribute(t, this->key);
  }

private:
  OperatorAttributeKey key;
};

tl::optional<OperatorAttributeValue>
    get_attribute(opmeta::OperatorParameters const &p,
                  OperatorAttributeKey key) {
  return mpark::visit(GetAttribute(key), p);
}

} // namespace substitutions
} // namespace FlexFlow
