#include "doctest/doctest.h"
#include "legion/legion_utilities.h"
#include "op-attrs/ffconst.h"
#include "serialization.h"
#include <rapidcheck.h>

using namespace FlexFlow;

TEST_CASE("Serialization") {
  Legion::Serializer sez;
  Legion::Deserializer dez(sez.get_buffer(), sez.get_buffer_size());

  using CompleteOperatorAttrs =
      variant_join<PCGOperatorAttrs, CompGraphOperatorAttrs>;

  std::vector<CompleteOperatorAttrs> operator_attrs {
    BatchMatmulAttrs batch_mm_attrs, BatchNormAttrs batch_norm_attrs,
        BroadcastAttrs broadcast_attrs, CastAttrs cast_attrs,
        CombineAttrs combine_attrs, ConcatAttrs concat_attrs,
        Conv2DAttrs conv2d_attrs, DropoutAttrs dropout_attrs,
        ElementBinaryAttrs elem_bin_attrs, ElementUnaryAttrs elem_unary_attrs,
        ElementScalarUnaryAttrs elem_scalar_unary_attrs,
        EmbeddingAttrs embedding_attrs, FlatAttrs flat_attrs,
        GatherAttrs gather_attrs, InputAttrs input_attrs,
        LayerNormAttrs layer_norm_attrs, LinearAttrs linear_attrs,
        MultiHeadAttentionAttrs mha_attrs, NoopAttrs noop_attrs,
        Pool2DAttrs pool2d_attrs, ReduceAttrs reduce_attrs,
        ReductionAttrs reduction_attrs, RepartitionAttrs repartition_attrs,
        ReplicateAttrs replicate_attrs, ReverseAttrs reverse_attrs,
        ReshapeAttrs reshape_attrs, SplitAttrs split_attrs,
        SoftmaxAttrs softmax_attrs, TopKAttrs topk_attrs,
        TransposeAttrs transpose_attrs
  }

  for (CompleteOperatorAttrs const &op : operator_attrs) {
    RC_SUBCASE("Serialization", [](CompleteOperatorAttrs const &pre_op) {
      pre_op = *rc::gen::arbitrary<CompleteOperatorAttrs>();
      auto post_op = pre_op;
      ff_task_serialize<CompleteOperatorAttrs>(sez, post_op);
      auto post_op = ff_task_deserialize<CompleteOperatorAttrs>(dez);
      RC_ASSERT(post_op == pre_op);
    });
  }
}
