// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/pcg_operator_attrs.variant.toml
/* proj-data
{
  "generated_from": "72d324ec59ca0c5a390458ea20e79338"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PCG_OPERATOR_ATTRS_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PCG_OPERATOR_ATTRS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/ops/attention_attrs.dtg.h"
#include "op-attrs/ops/batch_matmul.dtg.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include "op-attrs/ops/cast_attrs.dtg.h"
#include "op-attrs/ops/combine_attrs.dtg.h"
#include "op-attrs/ops/concat_attrs.dtg.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"
#include "op-attrs/ops/dropout_attrs.dtg.h"
#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/ops/flat_attrs.dtg.h"
#include "op-attrs/ops/gather_attrs.dtg.h"
#include "op-attrs/ops/input_attrs.dtg.h"
#include "op-attrs/ops/layer_norm_attrs.dtg.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/ops/noop_attrs.dtg.h"
#include "op-attrs/ops/pool_2d_attrs.dtg.h"
#include "op-attrs/ops/reduce_attrs.dtg.h"
#include "op-attrs/ops/reduction_attrs.dtg.h"
#include "op-attrs/ops/repartition_attrs.dtg.h"
#include "op-attrs/ops/replicate_attrs.dtg.h"
#include "op-attrs/ops/reshape_attrs.dtg.h"
#include "op-attrs/ops/reverse_attrs.dtg.h"
#include "op-attrs/ops/softmax_attrs.dtg.h"
#include "op-attrs/ops/split_attrs.dtg.h"
#include "op-attrs/ops/topk_attrs.dtg.h"
#include "op-attrs/ops/transpose_attrs.dtg.h"
#include "op-attrs/ops/weight_attrs.dtg.h"
#include "rapidcheck.h"
#include <cstddef>
#include <functional>
#include <ostream>
#include <type_traits>
#include <variant>

namespace FlexFlow {
struct PCGOperatorAttrs {
  PCGOperatorAttrs() = delete;
  explicit PCGOperatorAttrs(::FlexFlow::BatchMatmulAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::BatchNormAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::CastAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::CombineAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::ConcatAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::Conv2DAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::DropoutAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::ElementBinaryAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::ElementUnaryAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::EmbeddingAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::FlatAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::GatherAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::InputAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::LayerNormAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::LinearAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::MultiHeadAttentionAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::NoopAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::Pool2DAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::ReduceAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::ReductionAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::RepartitionAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::ReplicateAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::ReverseAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::ReshapeAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::SplitAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::SoftmaxAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::TopKAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::TransposeAttrs const &);
  explicit PCGOperatorAttrs(::FlexFlow::WeightAttrs const &);
  template <typename T>
  static constexpr bool IsPartOfPCGOperatorAttrs_v =
      std::is_same_v<T, ::FlexFlow::BatchMatmulAttrs> ||
      std::is_same_v<T, ::FlexFlow::BatchNormAttrs> ||
      std::is_same_v<T, ::FlexFlow::CastAttrs> ||
      std::is_same_v<T, ::FlexFlow::CombineAttrs> ||
      std::is_same_v<T, ::FlexFlow::ConcatAttrs> ||
      std::is_same_v<T, ::FlexFlow::Conv2DAttrs> ||
      std::is_same_v<T, ::FlexFlow::DropoutAttrs> ||
      std::is_same_v<T, ::FlexFlow::ElementBinaryAttrs> ||
      std::is_same_v<T, ::FlexFlow::ElementUnaryAttrs> ||
      std::is_same_v<T, ::FlexFlow::EmbeddingAttrs> ||
      std::is_same_v<T, ::FlexFlow::FlatAttrs> ||
      std::is_same_v<T, ::FlexFlow::GatherAttrs> ||
      std::is_same_v<T, ::FlexFlow::InputAttrs> ||
      std::is_same_v<T, ::FlexFlow::LayerNormAttrs> ||
      std::is_same_v<T, ::FlexFlow::LinearAttrs> ||
      std::is_same_v<T, ::FlexFlow::MultiHeadAttentionAttrs> ||
      std::is_same_v<T, ::FlexFlow::NoopAttrs> ||
      std::is_same_v<T, ::FlexFlow::Pool2DAttrs> ||
      std::is_same_v<T, ::FlexFlow::ReduceAttrs> ||
      std::is_same_v<T, ::FlexFlow::ReductionAttrs> ||
      std::is_same_v<T, ::FlexFlow::RepartitionAttrs> ||
      std::is_same_v<T, ::FlexFlow::ReplicateAttrs> ||
      std::is_same_v<T, ::FlexFlow::ReverseAttrs> ||
      std::is_same_v<T, ::FlexFlow::ReshapeAttrs> ||
      std::is_same_v<T, ::FlexFlow::SplitAttrs> ||
      std::is_same_v<T, ::FlexFlow::SoftmaxAttrs> ||
      std::is_same_v<T, ::FlexFlow::TopKAttrs> ||
      std::is_same_v<T, ::FlexFlow::TransposeAttrs> ||
      std::is_same_v<T, ::FlexFlow::WeightAttrs>;
  template <typename ReturnType, typename Visitor>
  ReturnType visit(Visitor &&v) const {
    switch (this->index()) {
      case 0: {
        ReturnType result = v(this->get<::FlexFlow::BatchMatmulAttrs>());
        return result;
      }
      case 1: {
        ReturnType result = v(this->get<::FlexFlow::BatchNormAttrs>());
        return result;
      }
      case 2: {
        ReturnType result = v(this->get<::FlexFlow::CastAttrs>());
        return result;
      }
      case 3: {
        ReturnType result = v(this->get<::FlexFlow::CombineAttrs>());
        return result;
      }
      case 4: {
        ReturnType result = v(this->get<::FlexFlow::ConcatAttrs>());
        return result;
      }
      case 5: {
        ReturnType result = v(this->get<::FlexFlow::Conv2DAttrs>());
        return result;
      }
      case 6: {
        ReturnType result = v(this->get<::FlexFlow::DropoutAttrs>());
        return result;
      }
      case 7: {
        ReturnType result = v(this->get<::FlexFlow::ElementBinaryAttrs>());
        return result;
      }
      case 8: {
        ReturnType result = v(this->get<::FlexFlow::ElementUnaryAttrs>());
        return result;
      }
      case 9: {
        ReturnType result = v(this->get<::FlexFlow::EmbeddingAttrs>());
        return result;
      }
      case 10: {
        ReturnType result = v(this->get<::FlexFlow::FlatAttrs>());
        return result;
      }
      case 11: {
        ReturnType result = v(this->get<::FlexFlow::GatherAttrs>());
        return result;
      }
      case 12: {
        ReturnType result = v(this->get<::FlexFlow::InputAttrs>());
        return result;
      }
      case 13: {
        ReturnType result = v(this->get<::FlexFlow::LayerNormAttrs>());
        return result;
      }
      case 14: {
        ReturnType result = v(this->get<::FlexFlow::LinearAttrs>());
        return result;
      }
      case 15: {
        ReturnType result = v(this->get<::FlexFlow::MultiHeadAttentionAttrs>());
        return result;
      }
      case 16: {
        ReturnType result = v(this->get<::FlexFlow::NoopAttrs>());
        return result;
      }
      case 17: {
        ReturnType result = v(this->get<::FlexFlow::Pool2DAttrs>());
        return result;
      }
      case 18: {
        ReturnType result = v(this->get<::FlexFlow::ReduceAttrs>());
        return result;
      }
      case 19: {
        ReturnType result = v(this->get<::FlexFlow::ReductionAttrs>());
        return result;
      }
      case 20: {
        ReturnType result = v(this->get<::FlexFlow::RepartitionAttrs>());
        return result;
      }
      case 21: {
        ReturnType result = v(this->get<::FlexFlow::ReplicateAttrs>());
        return result;
      }
      case 22: {
        ReturnType result = v(this->get<::FlexFlow::ReverseAttrs>());
        return result;
      }
      case 23: {
        ReturnType result = v(this->get<::FlexFlow::ReshapeAttrs>());
        return result;
      }
      case 24: {
        ReturnType result = v(this->get<::FlexFlow::SplitAttrs>());
        return result;
      }
      case 25: {
        ReturnType result = v(this->get<::FlexFlow::SoftmaxAttrs>());
        return result;
      }
      case 26: {
        ReturnType result = v(this->get<::FlexFlow::TopKAttrs>());
        return result;
      }
      case 27: {
        ReturnType result = v(this->get<::FlexFlow::TransposeAttrs>());
        return result;
      }
      case 28: {
        ReturnType result = v(this->get<::FlexFlow::WeightAttrs>());
        return result;
      }
      default: {
        throw std::runtime_error(fmt::format(
            "Unknown index {} for type PCGOperatorAttrs", this->index()));
      }
    }
  }
  template <typename ReturnType, typename Visitor>
  ReturnType visit(Visitor &&v) {
    switch (this->index()) {
      case 0: {
        ReturnType result = v(this->get<::FlexFlow::BatchMatmulAttrs>());
        return result;
      }
      case 1: {
        ReturnType result = v(this->get<::FlexFlow::BatchNormAttrs>());
        return result;
      }
      case 2: {
        ReturnType result = v(this->get<::FlexFlow::CastAttrs>());
        return result;
      }
      case 3: {
        ReturnType result = v(this->get<::FlexFlow::CombineAttrs>());
        return result;
      }
      case 4: {
        ReturnType result = v(this->get<::FlexFlow::ConcatAttrs>());
        return result;
      }
      case 5: {
        ReturnType result = v(this->get<::FlexFlow::Conv2DAttrs>());
        return result;
      }
      case 6: {
        ReturnType result = v(this->get<::FlexFlow::DropoutAttrs>());
        return result;
      }
      case 7: {
        ReturnType result = v(this->get<::FlexFlow::ElementBinaryAttrs>());
        return result;
      }
      case 8: {
        ReturnType result = v(this->get<::FlexFlow::ElementUnaryAttrs>());
        return result;
      }
      case 9: {
        ReturnType result = v(this->get<::FlexFlow::EmbeddingAttrs>());
        return result;
      }
      case 10: {
        ReturnType result = v(this->get<::FlexFlow::FlatAttrs>());
        return result;
      }
      case 11: {
        ReturnType result = v(this->get<::FlexFlow::GatherAttrs>());
        return result;
      }
      case 12: {
        ReturnType result = v(this->get<::FlexFlow::InputAttrs>());
        return result;
      }
      case 13: {
        ReturnType result = v(this->get<::FlexFlow::LayerNormAttrs>());
        return result;
      }
      case 14: {
        ReturnType result = v(this->get<::FlexFlow::LinearAttrs>());
        return result;
      }
      case 15: {
        ReturnType result = v(this->get<::FlexFlow::MultiHeadAttentionAttrs>());
        return result;
      }
      case 16: {
        ReturnType result = v(this->get<::FlexFlow::NoopAttrs>());
        return result;
      }
      case 17: {
        ReturnType result = v(this->get<::FlexFlow::Pool2DAttrs>());
        return result;
      }
      case 18: {
        ReturnType result = v(this->get<::FlexFlow::ReduceAttrs>());
        return result;
      }
      case 19: {
        ReturnType result = v(this->get<::FlexFlow::ReductionAttrs>());
        return result;
      }
      case 20: {
        ReturnType result = v(this->get<::FlexFlow::RepartitionAttrs>());
        return result;
      }
      case 21: {
        ReturnType result = v(this->get<::FlexFlow::ReplicateAttrs>());
        return result;
      }
      case 22: {
        ReturnType result = v(this->get<::FlexFlow::ReverseAttrs>());
        return result;
      }
      case 23: {
        ReturnType result = v(this->get<::FlexFlow::ReshapeAttrs>());
        return result;
      }
      case 24: {
        ReturnType result = v(this->get<::FlexFlow::SplitAttrs>());
        return result;
      }
      case 25: {
        ReturnType result = v(this->get<::FlexFlow::SoftmaxAttrs>());
        return result;
      }
      case 26: {
        ReturnType result = v(this->get<::FlexFlow::TopKAttrs>());
        return result;
      }
      case 27: {
        ReturnType result = v(this->get<::FlexFlow::TransposeAttrs>());
        return result;
      }
      case 28: {
        ReturnType result = v(this->get<::FlexFlow::WeightAttrs>());
        return result;
      }
      default: {
        throw std::runtime_error(fmt::format(
            "Unknown index {} for type PCGOperatorAttrs", this->index()));
      }
    }
  }
  template <typename T>
  bool has() const {
    static_assert(
        IsPartOfPCGOperatorAttrs_v<T>,
        "PCGOperatorAttrs::has() expected one of "
        "[::FlexFlow::BatchMatmulAttrs, ::FlexFlow::BatchNormAttrs, "
        "::FlexFlow::CastAttrs, ::FlexFlow::CombineAttrs, "
        "::FlexFlow::ConcatAttrs, ::FlexFlow::Conv2DAttrs, "
        "::FlexFlow::DropoutAttrs, ::FlexFlow::ElementBinaryAttrs, "
        "::FlexFlow::ElementUnaryAttrs, ::FlexFlow::EmbeddingAttrs, "
        "::FlexFlow::FlatAttrs, ::FlexFlow::GatherAttrs, "
        "::FlexFlow::InputAttrs, ::FlexFlow::LayerNormAttrs, "
        "::FlexFlow::LinearAttrs, ::FlexFlow::MultiHeadAttentionAttrs, "
        "::FlexFlow::NoopAttrs, ::FlexFlow::Pool2DAttrs, "
        "::FlexFlow::ReduceAttrs, ::FlexFlow::ReductionAttrs, "
        "::FlexFlow::RepartitionAttrs, ::FlexFlow::ReplicateAttrs, "
        "::FlexFlow::ReverseAttrs, ::FlexFlow::ReshapeAttrs, "
        "::FlexFlow::SplitAttrs, ::FlexFlow::SoftmaxAttrs, "
        "::FlexFlow::TopKAttrs, ::FlexFlow::TransposeAttrs, "
        "::FlexFlow::WeightAttrs], received T");
    return std::holds_alternative<T>(this->raw_variant);
  }
  template <typename T>
  T const &get() const {
    static_assert(
        IsPartOfPCGOperatorAttrs_v<T>,
        "PCGOperatorAttrs::get() expected one of "
        "[::FlexFlow::BatchMatmulAttrs, ::FlexFlow::BatchNormAttrs, "
        "::FlexFlow::CastAttrs, ::FlexFlow::CombineAttrs, "
        "::FlexFlow::ConcatAttrs, ::FlexFlow::Conv2DAttrs, "
        "::FlexFlow::DropoutAttrs, ::FlexFlow::ElementBinaryAttrs, "
        "::FlexFlow::ElementUnaryAttrs, ::FlexFlow::EmbeddingAttrs, "
        "::FlexFlow::FlatAttrs, ::FlexFlow::GatherAttrs, "
        "::FlexFlow::InputAttrs, ::FlexFlow::LayerNormAttrs, "
        "::FlexFlow::LinearAttrs, ::FlexFlow::MultiHeadAttentionAttrs, "
        "::FlexFlow::NoopAttrs, ::FlexFlow::Pool2DAttrs, "
        "::FlexFlow::ReduceAttrs, ::FlexFlow::ReductionAttrs, "
        "::FlexFlow::RepartitionAttrs, ::FlexFlow::ReplicateAttrs, "
        "::FlexFlow::ReverseAttrs, ::FlexFlow::ReshapeAttrs, "
        "::FlexFlow::SplitAttrs, ::FlexFlow::SoftmaxAttrs, "
        "::FlexFlow::TopKAttrs, ::FlexFlow::TransposeAttrs, "
        "::FlexFlow::WeightAttrs], received T");
    return std::get<T>(this->raw_variant);
  }
  template <typename T>
  T &get() {
    static_assert(
        IsPartOfPCGOperatorAttrs_v<T>,
        "PCGOperatorAttrs::get() expected one of "
        "[::FlexFlow::BatchMatmulAttrs, ::FlexFlow::BatchNormAttrs, "
        "::FlexFlow::CastAttrs, ::FlexFlow::CombineAttrs, "
        "::FlexFlow::ConcatAttrs, ::FlexFlow::Conv2DAttrs, "
        "::FlexFlow::DropoutAttrs, ::FlexFlow::ElementBinaryAttrs, "
        "::FlexFlow::ElementUnaryAttrs, ::FlexFlow::EmbeddingAttrs, "
        "::FlexFlow::FlatAttrs, ::FlexFlow::GatherAttrs, "
        "::FlexFlow::InputAttrs, ::FlexFlow::LayerNormAttrs, "
        "::FlexFlow::LinearAttrs, ::FlexFlow::MultiHeadAttentionAttrs, "
        "::FlexFlow::NoopAttrs, ::FlexFlow::Pool2DAttrs, "
        "::FlexFlow::ReduceAttrs, ::FlexFlow::ReductionAttrs, "
        "::FlexFlow::RepartitionAttrs, ::FlexFlow::ReplicateAttrs, "
        "::FlexFlow::ReverseAttrs, ::FlexFlow::ReshapeAttrs, "
        "::FlexFlow::SplitAttrs, ::FlexFlow::SoftmaxAttrs, "
        "::FlexFlow::TopKAttrs, ::FlexFlow::TransposeAttrs, "
        "::FlexFlow::WeightAttrs], received T");
    return std::get<T>(this->raw_variant);
  }
  size_t index() const {
    return this->raw_variant.index();
  }
  bool operator==(PCGOperatorAttrs const &) const;
  bool operator!=(PCGOperatorAttrs const &) const;
  bool operator<(PCGOperatorAttrs const &) const;
  bool operator>(PCGOperatorAttrs const &) const;
  bool operator<=(PCGOperatorAttrs const &) const;
  bool operator>=(PCGOperatorAttrs const &) const;
  std::variant<::FlexFlow::BatchMatmulAttrs,
               ::FlexFlow::BatchNormAttrs,
               ::FlexFlow::CastAttrs,
               ::FlexFlow::CombineAttrs,
               ::FlexFlow::ConcatAttrs,
               ::FlexFlow::Conv2DAttrs,
               ::FlexFlow::DropoutAttrs,
               ::FlexFlow::ElementBinaryAttrs,
               ::FlexFlow::ElementUnaryAttrs,
               ::FlexFlow::EmbeddingAttrs,
               ::FlexFlow::FlatAttrs,
               ::FlexFlow::GatherAttrs,
               ::FlexFlow::InputAttrs,
               ::FlexFlow::LayerNormAttrs,
               ::FlexFlow::LinearAttrs,
               ::FlexFlow::MultiHeadAttentionAttrs,
               ::FlexFlow::NoopAttrs,
               ::FlexFlow::Pool2DAttrs,
               ::FlexFlow::ReduceAttrs,
               ::FlexFlow::ReductionAttrs,
               ::FlexFlow::RepartitionAttrs,
               ::FlexFlow::ReplicateAttrs,
               ::FlexFlow::ReverseAttrs,
               ::FlexFlow::ReshapeAttrs,
               ::FlexFlow::SplitAttrs,
               ::FlexFlow::SoftmaxAttrs,
               ::FlexFlow::TopKAttrs,
               ::FlexFlow::TransposeAttrs,
               ::FlexFlow::WeightAttrs>
      raw_variant;
};
} // namespace FlexFlow
namespace std {
template <>
struct hash<::FlexFlow::PCGOperatorAttrs> {
  size_t operator()(::FlexFlow::PCGOperatorAttrs const &) const;
};
} // namespace std
namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::PCGOperatorAttrs> {
  static ::FlexFlow::PCGOperatorAttrs from_json(json const &);
  static void to_json(json &, ::FlexFlow::PCGOperatorAttrs const &);
};
} // namespace nlohmann
namespace rc {
template <>
struct Arbitrary<::FlexFlow::PCGOperatorAttrs> {
  static Gen<::FlexFlow::PCGOperatorAttrs> arbitrary();
};
} // namespace rc
namespace FlexFlow {
std::string format_as(::FlexFlow::PCGOperatorAttrs const &);
std::ostream &operator<<(std::ostream &, ::FlexFlow::PCGOperatorAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PCG_OPERATOR_ATTRS_DTG_H