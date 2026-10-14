#include "substitutions/output_graph/materialize_operator_from_attrs_map.h"
#include "utils/containers/contains_key.h"
#include "utils/fmt/map.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

struct Accessor {
  Accessor(std::map<OperatorAttributeKey, OperatorAttributeValue> const &m)
      : m(m) {}

  std::map<OperatorAttributeKey, OperatorAttributeValue> const &m;

  template <typename T>
  T get(OperatorAttributeKey k) const {
    if (contains_key(this->m, k)) {
      return this->m.at(k).get<T>();
    } else {
      PANIC("Could not find key in attrs map", k, this->m);
    }
  }

  positive_int get_positive_int(OperatorAttributeKey k) const {
    return positive_int{
        this->get<nonnegative_int>(k),
    };
  }
};

PCGOperatorAttrs materialize_operator_from_attrs_map(
    std::map<OperatorAttributeKey, OperatorAttributeValue> const &attrs) {
  OperatorType op_type =
      attrs.at(OperatorAttributeKey::OP_TYPE).get<OperatorType>();

  Accessor acc = Accessor{attrs};

  switch (op_type) {
    case OperatorType::BATCHNORM:
      return PCGOperatorAttrs{
          BatchNormAttrs{
              /*activation=*/acc.get<std::optional<Activation>>(
                  OperatorAttributeKey::ACTIVATION),
              /*affine=*/acc.get<bool>(OperatorAttributeKey::AFFINE),
              /*eps=*/acc.get<float>(OperatorAttributeKey::EPSILON),
              /*momentum=*/
              acc.get<std::optional<float>>(OperatorAttributeKey::MOMENTUM),
          },
      };
    case OperatorType::MULTIHEAD_ATTENTION:
      return PCGOperatorAttrs{MultiHeadAttentionAttrs{
          /*embed_dim=*/acc.get_positive_int(OperatorAttributeKey::EMBED_DIM),
          /*num_heads=*/
          acc.get_positive_int(OperatorAttributeKey::NUM_HEADS),
          /*kdim=*/acc.get_positive_int(OperatorAttributeKey::KDIM),
          /*vdim=*/acc.get_positive_int(OperatorAttributeKey::VDIM),
          /*dropout=*/acc.get<float>(OperatorAttributeKey::DROPOUT),
          /*bias=*/acc.get<bool>(OperatorAttributeKey::BIAS),
          /*add_bias_kv=*/acc.get<bool>(OperatorAttributeKey::ADD_BIAS_KV),
          /*add_zero_attn=*/acc.get<bool>(OperatorAttributeKey::ADD_ZERO_ATTN),
      }};
    case OperatorType::POOL2D:
      return PCGOperatorAttrs{Pool2DAttrs{
          /*kernel_h=*/acc.get_positive_int(OperatorAttributeKey::KERNEL_H),
          /*kernel_w=*/acc.get_positive_int(OperatorAttributeKey::KERNEL_W),
          /*stride_h=*/acc.get_positive_int(OperatorAttributeKey::STRIDE_H),
          /*stride_w=*/acc.get_positive_int(OperatorAttributeKey::STRIDE_W),
          /*padding_h=*/
          acc.get<nonnegative_int>(OperatorAttributeKey::PADDING_H),
          /*padding_w=*/
          acc.get<nonnegative_int>(OperatorAttributeKey::PADDING_W),
          /*pool_type=*/acc.get<PoolOp>(OperatorAttributeKey::POOL_TYPE),
          /*activation=*/
          acc.get<std::optional<Activation>>(OperatorAttributeKey::ACTIVATION)
              .value(),
      }};
    case OperatorType::NOOP:
    case OperatorType::INPUT:
    case OperatorType::WEIGHT:
    case OperatorType::DROPOUT:
    case OperatorType::LINEAR:
      return PCGOperatorAttrs{LinearAttrs{
          /*out_channels=*/acc.get_positive_int(
              OperatorAttributeKey::OUT_CHANNELS),
          /*use_bias=*/acc.get<bool>(OperatorAttributeKey::USE_BIAS),
          /*data_type=*/acc.get<DataType>(OperatorAttributeKey::DATA_TYPE),
          /*activation=*/
          acc.get<std::optional<Activation>>(OperatorAttributeKey::ACTIVATION),
          /*regularizer=*/
          acc.get<std::optional<RegularizerAttrs>>(
              OperatorAttributeKey::REGULARIZER),
      }};
    case OperatorType::CONV2D:
      return PCGOperatorAttrs{Conv2DAttrs{
          /*out_channels=*/acc.get_positive_int(
              OperatorAttributeKey::OUT_CHANNELS),
          /*kernel_h=*/acc.get_positive_int(OperatorAttributeKey::KERNEL_H),
          /*kernel_w=*/acc.get_positive_int(OperatorAttributeKey::KERNEL_W),
          /*stride_h=*/acc.get_positive_int(OperatorAttributeKey::STRIDE_H),
          /*stride_w=*/acc.get_positive_int(OperatorAttributeKey::STRIDE_W),
          /*padding_h=*/
          acc.get<nonnegative_int>(OperatorAttributeKey::PADDING_H),
          /*padding_w=*/
          acc.get<nonnegative_int>(OperatorAttributeKey::PADDING_W),
          /*groups=*/acc.get_positive_int(OperatorAttributeKey::GROUPS),
          /*activation=*/
          acc.get<std::optional<Activation>>(OperatorAttributeKey::ACTIVATION),
          /*use_bias=*/acc.get<bool>(OperatorAttributeKey::USE_BIAS),
      }};
    case OperatorType::RELU:
      return PCGOperatorAttrs{ElementUnaryAttrs{
          acc.get<ElementUnaryOp>(OperatorAttributeKey::SUB_OP),
          acc.get<std::optional<float>>(OperatorAttributeKey::SCALAR),
      }};
    case OperatorType::SOFTMAX:
      return PCGOperatorAttrs{SoftmaxAttrs{
          acc.get<ff_dim_t>(OperatorAttributeKey::AXIS),
      }};
    case OperatorType::EW_ADD:
      return PCGOperatorAttrs{ElementBinaryAttrs{
          acc.get<ElementBinaryOp>(OperatorAttributeKey::SUB_OP),
          acc.get<DataType>(OperatorAttributeKey::DATA_TYPE),
          acc.get<bool>(OperatorAttributeKey::SHOULD_BROADCAST_LHS),
          acc.get<bool>(OperatorAttributeKey::SHOULD_BROADCAST_LHS),
      }};
    case OperatorType::REPLICATE:
      return PCGOperatorAttrs{ReplicateAttrs{
          /*replicate_degree=*/acc.get_positive_int(
              OperatorAttributeKey::PARALLEL_DEGREE),
      }};
    case OperatorType::REPARTITION:
      return PCGOperatorAttrs{RepartitionAttrs{
          /*repartition_dim=*/acc.get<ff_dim_t>(
              OperatorAttributeKey::PARALLEL_DIM),
          /*repartition_Degree=*/
          acc.get_positive_int(OperatorAttributeKey::PARALLEL_DEGREE),
      }};
    case OperatorType::COMBINE:
      return PCGOperatorAttrs{CombineAttrs{
          /*combine_dim=*/acc.get<ff_dim_t>(OperatorAttributeKey::PARALLEL_DIM),
          /*combine_degree=*/
          acc.get_positive_int(OperatorAttributeKey::PARALLEL_DEGREE),
      }};
    case OperatorType::REDUCTION:
      return PCGOperatorAttrs{ReductionAttrs{
          acc.get_positive_int(OperatorAttributeKey::PARALLEL_DEGREE),
      }};
    case OperatorType::BATCHMATMUL:
    case OperatorType::SCALAR_MULTIPLY:
    case OperatorType::SCALAR_ADD:
    case OperatorType::SCALAR_FLOOR_DIV:
    case OperatorType::SCALAR_TRUE_DIV:
    case OperatorType::SCALAR_SUB:
    case OperatorType::IDENTITY:
    case OperatorType::SIGMOID:
    case OperatorType::TANH:
    case OperatorType::ELU:
    case OperatorType::FLAT:
    case OperatorType::CONCAT:
    case OperatorType::SPLIT:
    case OperatorType::EMBEDDING:
    case OperatorType::CACHE:
    case OperatorType::RESHAPE:
    case OperatorType::REVERSE:
    case OperatorType::TRANSPOSE:
    case OperatorType::EW_MUL:
    case OperatorType::MATMUL:
    case OperatorType::MUL:
    case OperatorType::ENLARGE:
    case OperatorType::SQUEEZE:
    case OperatorType::UNSQUEEZE:
    case OperatorType::EW_SUB:
    case OperatorType::EW_DIV:
    case OperatorType::EW_EQUAL:
    case OperatorType::EW_GREATER:
    case OperatorType::EW_LESS:
    case OperatorType::EW_MAX:
    case OperatorType::EW_MIN:
    case OperatorType::REDUCE_ARGMAX:
    case OperatorType::REDUCE_ARGMIN:
    case OperatorType::REDUCE_MAX:
    case OperatorType::REDUCE_MEAN:
    case OperatorType::REDUCE_MIN:
    case OperatorType::REDUCE_PROD:
    case OperatorType::REDUCE_SUM:
    case OperatorType::PAD:
    case OperatorType::SHAPE:
    case OperatorType::SIZE:
    case OperatorType::TOPK:
    case OperatorType::WHERE:
    case OperatorType::CEIL:
    case OperatorType::CAST:
    case OperatorType::EXP:
    case OperatorType::ROUND:
    case OperatorType::LOG:
    case OperatorType::LOGICAL_NOT:
    case OperatorType::SQRT:
    case OperatorType::SIN:
    case OperatorType::COS:
    case OperatorType::LEAKYRELU:
    case OperatorType::SLICE:
    case OperatorType::RESIZE:
    case OperatorType::PRELU:
    case OperatorType::GELU:
    case OperatorType::FUSED:
    case OperatorType::RSQRT:
    case OperatorType::POW:
    case OperatorType::MEAN:
    case OperatorType::LAYERNORM:
    case OperatorType::GATHER:
    case OperatorType::BROADCAST:
    case OperatorType::BATCH:
    case OperatorType::PIPELINE:
    case OperatorType::FUSED_PARALLEL:
    default:
      PANIC("Unsupported operator type", op_type);
  }
}

} // namespace FlexFlow
