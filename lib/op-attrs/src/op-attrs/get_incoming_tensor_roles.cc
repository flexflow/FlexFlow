#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "utils/overload.h"

namespace FlexFlow {

std::vector<IncomingTensorRole> get_incoming_tensor_roles(
    ComputationGraphOpAttrs const &comp_graph_op_attrs, int num_incoming) {
  return get_incoming_tensor_roles(
      pcg_op_attrs_from_compgraph_op_attrs(comp_graph_op_attrs), num_incoming);
}

std::vector<IncomingTensorRole>
    get_incoming_tensor_roles(PCGOperatorAttrs const &pcg_op_attrs,
                              int num_incoming) {
  return pcg_op_attrs.visit<std::vector<IncomingTensorRole>>(overload{
      [](BatchMatmulAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT,
                           IncomingTensorRole::INPUT};
      },
      [](BatchNormAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](BroadcastAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](CastAttrs const &) { return std::vector{IncomingTensorRole::INPUT}; },
      [](CombineAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [&](ConcatAttrs const &) {
        return std::vector(num_incoming, IncomingTensorRole::INPUT);
      },
      [](Conv2DAttrs const &attrs) {
        return get_conv2d_incoming_tensor_roles(attrs);
      },
      [](DropoutAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](ElementBinaryAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT,
                           IncomingTensorRole::INPUT};
      },
      [](ElementUnaryAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](EmbeddingAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT,
                           IncomingTensorRole::WEIGHT};
      },
      [](FlatAttrs const &) { return std::vector{IncomingTensorRole::INPUT}; },
      [](GatherAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](InputAttrs const &) { return std::vector<IncomingTensorRole>{}; },
      [](LayerNormAttrs const &attrs) {
        return get_layer_norm_incoming_tensor_roles(attrs);
      },
      [](LinearAttrs const &attrs) {
        return get_linear_incoming_tensor_roles(attrs);
      },
      [](MultiHeadAttentionAttrs const &attrs) {
        return get_attention_incoming_tensor_roles(attrs);
      },
      [](NoopAttrs const &) { return std::vector{IncomingTensorRole::INPUT}; },
      [](Pool2DAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](ReduceAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](ReductionAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](RepartitionAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](ReplicateAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](ReverseAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](ReshapeAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](SplitAttrs const &) { return std::vector{IncomingTensorRole::INPUT}; },
      [](SoftmaxAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](TopKAttrs const &) { return std::vector{IncomingTensorRole::INPUT}; },
      [](TransposeAttrs const &) {
        return std::vector{IncomingTensorRole::INPUT};
      },
      [](WeightAttrs const &) { return std::vector<IncomingTensorRole>{}; },
  });
}

} // namespace FlexFlow
