#include "substitutions/get_attribute.h"

namespace FlexFlow {

optional<OperatorAttributeValue> get_attribute(AggregateAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(AggregateSpecAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(BatchMatmulAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(CastAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::DATA_TYPE:
      return p.dtype;
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(CombineAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return p.combine_dim;
    case OperatorAttributeKey::PARALLEL_DIM:
      return p.combine_degree;
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(ConcatAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::AXIS:
      return p.axis;
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(Conv2DAttrs const &p,
                                               OperatorAttributeKey key) {
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
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(ElementBinaryAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(ElementUnaryAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(DropoutAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(EmbeddingAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::DATA_TYPE:
      return p.data_type;
    case OperatorAttributeKey::AGGR:
      return p.aggr;
    case OperatorAttributeKey::NUM_ENTRIES:
      return p.num_entries;
    case OperatorAttributeKey::OUT_CHANNELS:
      return p.out_channels;
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(FlatAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(GatherAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::AXIS:
      return p.dim;
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(Group_byAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(LayerNormAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(LinearAttrs const &p,
                                               OperatorAttributeKey key) {
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
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(MultiHeadAttentionAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::NUM_HEADS:
      return p.num_heads;
    case OperatorAttributeKey::USE_BIAS:
      return p.bias;
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(Pool2DAttrs const &p,
                                               OperatorAttributeKey key) {
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
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(ReduceAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(ReductionAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return p.reduction_dim;
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return p.reduction_degree;
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(RepartitionAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return p.repartition_dim;
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return p.repartition_degree;
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(ReplicateAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return p.replicate_dim;
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return p.replicate_degree;
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(ReshapeAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(SplitAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::AXIS:
      return p.axis;
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(SoftmaxAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::AXIS:
      return p.dim;
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(TopKAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    default:
      return nullopt;
  }
}

optional<OperatorAttributeValue> get_attribute(TransposeAttrs const &p,
                                               OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::PERMUTATION:
      return p.perm;
    default:
      return nullopt;
  }
}

struct GetAttribute {
  GetAttribute(OperatorAttributeKey key) : key(key) {}

  template <typename T>
  optional<OperatorAttributeValue> operator()(T const &t) {
    return get_attribute(t, this->key);
  }

private:
  OperatorAttributeKey key;
};

optional<OperatorAttributeValue> get_attribute(PCGOperatorAttrs const &p,
                                               OperatorAttributeKey key) {
  return visit(GetAttribute(key), p);
}

} // namespace FlexFlow
