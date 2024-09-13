#include "op-attrs/get_output_shapes.h"
#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ops/cast.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/concat.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/dropout.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/flat.h"
#include "op-attrs/ops/gather.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/ops/weight.h"
#include "utils/overload.h"

namespace FlexFlow {

std::vector<ParallelTensorShape>
    get_output_shapes(PCGOperatorAttrs const &pcg_op_attrs,
                      std::vector<ParallelTensorShape> const &inputs) {
  return pcg_op_attrs.visit<std::vector<ParallelTensorShape>>(overload{
      [&](BatchMatmulAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, inputs.at(0), inputs.at(1)))};
      },
      [&](BatchNormAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_shape(attrs, inputs.at(0))};
      },
      [&](CastAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(get_output_shape(attrs, inputs.at(0)))};
      },
      [&](CombineAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(get_output_shape(attrs, inputs.at(0)))};
      },
      [&](ConcatAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_shape(attrs, inputs)};
      },
      [&](Conv2DAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_shape(attrs, inputs.at(0))};
      },
      [&](DropoutAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(get_output_shape(attrs, inputs.at(0)))};
      },
      [&](ElementBinaryAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, inputs.at(0), inputs.at(1)))};
      },
      [&](ElementUnaryAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(get_output_shape(attrs, inputs.at(0)))};
      },
      [&](EmbeddingAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(get_output_shape(attrs, inputs.at(0)))};
      },
      [&](FlatAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_shape(attrs, inputs.at(0))};
      },
      [&](GatherAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_shape(attrs, inputs.at(0), inputs.at(1))};
      },
      [&](InputAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_parallel_tensor_shape(attrs)};
      },
      [&](LayerNormAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(get_output_shape(attrs, inputs.at(0)))};
      },
      [&](LinearAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(get_output_shape(attrs, inputs.at(0)))};
      },
      [&](ReplicateAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_shape(attrs, inputs.at(0))};
      },
      [&](WeightAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_parallel_tensor_shape(attrs)};
      },
      [&](auto const &attrs) -> std::vector<ParallelTensorShape> {
        NOT_IMPLEMENTED();
      }});
}

} // namespace FlexFlow
