#include "substitutions/operator_pattern/get_attribute.h"
#include "op-attrs/get_op_type.h"
#include "utils/containers/as_vector.h"

namespace FlexFlow {

std::optional<OperatorAttributeValue> get_attribute(BatchMatmulAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(BatchNormAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::RELU:
      return p.relu;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(BroadcastAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::TARGET_DIMS:
      return p.target_dims;
    default:
      return std::nullopt;
  }
}


std::optional<OperatorAttributeValue> get_attribute(CastAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::DATA_TYPE:
      return p.dtype;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(CombineAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return p.combine_dim;
    case OperatorAttributeKey::PARALLEL_DIM:
      return p.combine_degree;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ConcatAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::AXIS:
      return p.axis;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(Conv2DAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
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
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ElementBinaryAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ElementUnaryAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(DropoutAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(EmbeddingAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::DATA_TYPE:
      return p.data_type;
    case OperatorAttributeKey::AGGR:
      return p.aggr;
    case OperatorAttributeKey::NUM_ENTRIES:
      return p.num_entries;
    case OperatorAttributeKey::OUT_CHANNELS:
      return p.out_channels;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(FlatAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(GatherAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::AXIS:
      return p.dim;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(InputAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(LayerNormAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(LinearAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::OUT_CHANNELS:
      return p.out_channels;
    case OperatorAttributeKey::USE_BIAS:
      return p.use_bias;
    case OperatorAttributeKey::DATA_TYPE:
      return p.data_type;
    case OperatorAttributeKey::ACTIVATION:
      return p.activation;
    case OperatorAttributeKey::REGULARIZER:
      return p.regularizer;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue>
    get_attribute(MultiHeadAttentionAttrs const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::NUM_HEADS:
      return p.num_heads;
    case OperatorAttributeKey::USE_BIAS:
      return p.bias;
    case OperatorAttributeKey::DROPOUT:
      return p.dropout;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(NoopAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(Pool2DAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
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
      return std::optional<Activation>{p.activation};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ReduceAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ReductionAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return p.reduction_degree;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(RepartitionAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return p.repartition_dim;
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return p.repartition_degree;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ReplicateAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return p.replicate_degree;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ReshapeAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ReverseAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::AXIS:
      return p.axis;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(SplitAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::AXIS:
      return p.axis;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(SoftmaxAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::AXIS:
      return p.dim;
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(TopKAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(TransposeAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    case OperatorAttributeKey::PERMUTATION:
      return as_vector(p.perm);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(WeightAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return get_op_type(p);
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(PCGOperatorAttrs const &p,
                                                    OperatorAttributeKey key) {
  return p.visit<std::optional<OperatorAttributeValue>>(
      [&](auto const &attrs) { return get_attribute(attrs, key); });
}

} // namespace FlexFlow
