#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/attention/multihead_attention_inputs.h"
#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/extend.h"
#include "utils/expected.h"
#include "utils/integer_conversions.h"
#include "utils/not_implemented.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

/* bool MultiHeadAttentionAttrs::is_valid(std::vector<ParallelTensorShape> const
 * &inputs) const { */
/*   return (inputs.size() == 3 && std::all_of(inputs.begin(), inputs.end(),
 * [](ParallelTensorShape const &s) { return s.is_valid(); })); */
/*   bool is_valid = true; */
/*   return is_valid; */
/* } */

positive_int get_qProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.kdim;
}

positive_int get_vProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.vdim;
}

positive_int get_kProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.kdim;
}

positive_int get_oProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.embed_dim;
}

positive_int get_qSize(TensorShape const &query_shape) {
  return dim_at_idx(query_shape.dims, relative_ff_dim_t{0});
}

positive_int get_kSize(TensorShape const &key_shape) {
  return dim_at_idx(key_shape.dims, relative_ff_dim_t{0});
}

positive_int get_vSize(TensorShape const &value_shape) {
  return dim_at_idx(value_shape.dims, relative_ff_dim_t{0});
}

positive_int get_qSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.query_dim.size;
}

positive_int get_qSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.query_size;
}

positive_int get_kSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.key_dim.size;
}

positive_int get_kSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.key_size;
}

positive_int get_vSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.value_dim.size;
}

positive_int get_vSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.value_size;
}

positive_int get_kvSeqLength(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.sequence_dim.size;
}

positive_int get_kvSeqLength(MultiHeadAttentionInputs const &inputs) {
  return inputs.sequence_length;
}

positive_int get_qoSeqLength(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.sequence_dim.size; // FIXME -- assumes only prefill
}

positive_int get_qoSeqLength(MultiHeadAttentionInputs const &inputs) {
  return inputs.sequence_length; // FIXME -- assumes only prefil
}

positive_int get_num_samples(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.batch_dim.size;
}

positive_int get_num_samples(MultiHeadAttentionInputs const &inputs) {
  return inputs.batch_size;
}

static void check_attrs(MultiHeadAttentionAttrs const &attrs) {
  ASSERT(!attrs.add_bias_kv,
         "add_bias_kv is not yet supported. If you need this "
         "functionality, please create an issue.");
}

std::map<TensorSlotName, IncomingTensorRole>
    get_attention_incoming_tensor_roles(MultiHeadAttentionAttrs const &attrs) {

  check_attrs(attrs);

  std::map<TensorSlotName, IncomingTensorRole> roles = {
      {TensorSlotName::QUERY, IncomingTensorRole::INPUT},
      {TensorSlotName::KEY, IncomingTensorRole::INPUT},
      {TensorSlotName::VALUE, IncomingTensorRole::INPUT},
      {TensorSlotName::WEIGHT, IncomingTensorRole::WEIGHT},
  };

  if (attrs.bias) {
    roles[TensorSlotName::INPUT_BIAS] = IncomingTensorRole::WEIGHT;
    roles[TensorSlotName::OUTPUT_BIAS] = IncomingTensorRole::WEIGHT;
  }

  return roles;
}

tl::expected<TensorShape, std::string>
    get_output_shape(MultiHeadAttentionAttrs const &attrs,
                     TensorShape const &input_q,
                     TensorShape const &input_k,
                     TensorShape const &input_v) {
  check_attrs(attrs);

  tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
      parse_attention_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }

  MultiHeadAttentionInputs parsed = parse_result.value();

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          parsed.batch_size,
          parsed.sequence_length,
          attrs.embed_dim,
      }},
      parsed.datatype,
  };
}

tl::expected<TensorShape, std::string>
    get_weights_shape(MultiHeadAttentionAttrs const &attrs,
                      TensorShape const &input_q,
                      TensorShape const &input_k,
                      TensorShape const &input_v) {
  check_attrs(attrs);

  tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
      parse_attention_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }

  MultiHeadAttentionInputs parsed = parse_result.value();

  // W^Q_i in "Attention Is All You Need" top of page 5
  positive_int qProjectWeightSize = parsed.query_size * attrs.kdim;

  // W^K_i in "Attention Is All You Need" top of page 5 (all i's put together)
  positive_int kProjectWeightSize = parsed.key_size * attrs.kdim;

  // W^V_i in "Attention Is All You Need" top of page 5 (all i's put together)
  positive_int vProjectWeightSize = parsed.value_size * attrs.vdim;

  // W^O in "Attention Is All You Need" top of page 5, with num_heads factored
  // out
  positive_int outWeightSize = attrs.vdim * attrs.embed_dim;

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          (qProjectWeightSize + kProjectWeightSize + vProjectWeightSize +
           outWeightSize),
          attrs.num_heads,
      }},
      parsed.datatype,
  };
}

tl::expected<TensorShape, std::string>
    get_input_bias_shape(MultiHeadAttentionAttrs const &attrs,
                         TensorShape const &input_q,
                         TensorShape const &input_k,
                         TensorShape const &input_v) {
  check_attrs(attrs);

  MultiHeadAttentionInputs parsed = ({
    tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
        parse_attention_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }
    parse_result.value();
  });

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          attrs.kdim + attrs.kdim + attrs.vdim,
      }},
      parsed.datatype,
  };
}

tl::expected<TensorShape, std::string>
    get_output_bias_shape(MultiHeadAttentionAttrs const &attrs,
                          TensorShape const &input_q,
                          TensorShape const &input_k,
                          TensorShape const &input_v) {
  check_attrs(attrs);

  MultiHeadAttentionInputs parsed = ({
    tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
        parse_attention_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }
    parse_result.value();
  });

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          attrs.embed_dim,
      }},
      parsed.datatype,
  };
}

tl::expected<std::map<TensorSlotName, TensorShape>, std::string>
    get_weight_shapes(MultiHeadAttentionAttrs const &attrs,
                      TensorShape const &input_q,
                      TensorShape const &input_k,
                      TensorShape const &input_v) {

  std::map<TensorSlotName, TensorShape> weight_shapes = {
      {
          TensorSlotName::WEIGHT,
          PROPAGATE_ERR(get_weights_shape(attrs, input_q, input_k, input_v)),
      },
  };

  if (attrs.bias) {
    weight_shapes.insert({
        TensorSlotName::INPUT_BIAS,
        PROPAGATE_ERR(get_input_bias_shape(attrs, input_q, input_k, input_v)),
    });

    weight_shapes.insert({
        TensorSlotName::OUTPUT_BIAS,
        PROPAGATE_ERR(get_output_bias_shape(attrs, input_q, input_k, input_v)),
    });
  }

  return weight_shapes;
}

tl::expected<ParallelTensorShape, std::string>
    get_weights_shape(MultiHeadAttentionAttrs const &attrs,
                      ParallelTensorShape const &input_q,
                      ParallelTensorShape const &input_k,
                      ParallelTensorShape const &input_v) {
  check_attrs(attrs);

  tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
      parse_attention_parallel_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }
  MultiHeadAttentionParallelInputs parsed = parse_result.value();

  tl::expected<TensorShape, std::string> result_unpar_get_shape =
      get_weights_shape(attrs,
                        get_reduced_shape(input_q),
                        get_reduced_shape(input_k),
                        get_reduced_shape(input_v));
  if (!result_unpar_get_shape.has_value()) {
    return tl::unexpected(result_unpar_get_shape.error());
  }
  TensorShape unpar_shape = result_unpar_get_shape.value();

  positive_int joined_dim_degree = 1_p;
  positive_int head_dim_degree = parsed.discard_copy_degree.value;

  return lift_shape_to_parallel_with_degrees(
      unpar_shape,
      SumDegree{1_p},
      DiscardCopyDegree{parsed.batch_dim.degree},
      FFOrdered<positive_int>{joined_dim_degree, head_dim_degree});
}

tl::expected<ParallelTensorShape, std::string>
    get_input_bias_shape(MultiHeadAttentionAttrs const &attrs,
                         ParallelTensorShape const &input_q,
                         ParallelTensorShape const &input_k,
                         ParallelTensorShape const &input_v) {
  check_attrs(attrs);

  MultiHeadAttentionParallelInputs parsed = ({
    tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
        parse_attention_parallel_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }

    parse_result.value();
  });

  TensorShape unpar_shape = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_input_bias_shape(attrs,
                             get_reduced_shape(input_q),
                             get_reduced_shape(input_k),
                             get_reduced_shape(input_v));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }

    result_unpar.value();
  });

  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      parsed.batch_dim.degree * parsed.discard_copy_degree.value};
  FFOrdered<positive_int> shard_degrees = FFOrdered<positive_int>{1_p};
  return lift_shape_to_parallel_with_degrees(
      unpar_shape, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_output_bias_shape(MultiHeadAttentionAttrs const &attrs,
                          ParallelTensorShape const &input_q,
                          ParallelTensorShape const &input_k,
                          ParallelTensorShape const &input_v) {
  check_attrs(attrs);

  MultiHeadAttentionParallelInputs parsed = ({
    tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
        parse_attention_parallel_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }

    parse_result.value();
  });

  TensorShape unpar_shape = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_output_bias_shape(attrs,
                              get_reduced_shape(input_q),
                              get_reduced_shape(input_k),
                              get_reduced_shape(input_v));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }

    result_unpar.value();
  });

  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      parsed.batch_dim.degree * parsed.discard_copy_degree.value};
  FFOrdered<positive_int> shard_degrees = FFOrdered<positive_int>{1_p};
  return lift_shape_to_parallel_with_degrees(
      unpar_shape, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(MultiHeadAttentionAttrs const &attrs,
                     ParallelTensorShape const &input_q,
                     ParallelTensorShape const &input_k,
                     ParallelTensorShape const &input_v) {
  check_attrs(attrs);

  tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
      parse_attention_parallel_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }
  MultiHeadAttentionParallelInputs parsed = parse_result.value();

  tl::expected<TensorShape, std::string> result_unpar_get_shape =
      get_output_shape(attrs,
                       get_reduced_shape(input_q),
                       get_reduced_shape(input_k),
                       get_reduced_shape(input_v));
  if (!result_unpar_get_shape.has_value()) {
    return tl::unexpected(result_unpar_get_shape.error());
  }
  TensorShape unpar_shape = result_unpar_get_shape.value();

  positive_int sum_degree = parsed.discard_copy_degree.value;
  positive_int discard_copy_degree = 1_p;
  positive_int batch_degree = parsed.batch_dim.degree;
  positive_int seq_len_degree = 1_p;
  positive_int out_dim_degree = 1_p;

  return lift_shape_to_parallel_with_degrees(
      unpar_shape,
      SumDegree{sum_degree},
      DiscardCopyDegree{discard_copy_degree},
      FFOrdered{batch_degree, seq_len_degree, out_dim_degree});
}

positive_int get_oSize(ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

positive_int get_oSize(TensorShape const &) {
  NOT_IMPLEMENTED();
}

tl::expected<std::map<TensorSlotName, ParallelTensorShape>, std::string>
    get_weight_shapes(MultiHeadAttentionAttrs const &attrs,
                      ParallelTensorShape const &input_q,
                      ParallelTensorShape const &input_k,
                      ParallelTensorShape const &input_v) {

  std::map<TensorSlotName, ParallelTensorShape> weight_shapes = {
      {
          TensorSlotName::WEIGHT,
          PROPAGATE_ERR(get_weights_shape(attrs, input_q, input_k, input_v)),
      },
  };

  if (attrs.bias) {
    weight_shapes.insert({
        TensorSlotName::INPUT_BIAS,
        PROPAGATE_ERR(get_input_bias_shape(attrs, input_q, input_k, input_v)),
    });

    weight_shapes.insert({
        TensorSlotName::OUTPUT_BIAS,
        PROPAGATE_ERR(get_output_bias_shape(attrs, input_q, input_k, input_v)),
    });
  }

  return weight_shapes;
}

tl::expected<std::map<TensorSlotName, InitializerAttrs>, std::string>
    get_initializers(
        MultiHeadAttentionAttrs const &attrs,
        TensorShape const &input_q,
        TensorShape const &input_k,
        TensorShape const &input_v,
        std::optional<InitializerAttrs> const &maybe_weights_initializer,
        std::optional<InitializerAttrs> const &maybe_input_bias_initializer,
        std::optional<InitializerAttrs> const &maybe_output_bias_initializer) {
  check_attrs(attrs);

  if (!attrs.bias && maybe_input_bias_initializer.has_value()) {
    return tl::unexpected(
        fmt::format("Expected input_bias_initializer=std::nullopt since "
                    "bias=false, but received input_bias_initializer: {}",
                    maybe_input_bias_initializer.value()));
  }

  if (!attrs.bias && maybe_output_bias_initializer.has_value()) {
    return tl::unexpected(
        fmt::format("Expected output_bias_initializer=std::nullopt since "
                    "bias=false, but received output_bias_initializer: {}",
                    maybe_output_bias_initializer.value()));
  }

  InitializerAttrs default_weights_initializer = InitializerAttrs{
      GlorotUniformAttrs{
          /*seed=*/0,
      },
  };

  InitializerAttrs default_input_bias_initializer = InitializerAttrs{
      ZeroInitializerAttrs{},
  };

  InitializerAttrs default_output_bias_initializer = InitializerAttrs{
      ZeroInitializerAttrs{},
  };

  InitializerAttrs weights_initializer =
      maybe_weights_initializer.value_or(default_weights_initializer);
  InitializerAttrs input_bias_initializer =
      maybe_input_bias_initializer.value_or(default_input_bias_initializer);
  InitializerAttrs output_bias_initializer =
      maybe_output_bias_initializer.value_or(default_output_bias_initializer);

  if (attrs.bias) {
    return std::map<TensorSlotName, InitializerAttrs>{
        {TensorSlotName::WEIGHT, weights_initializer},
        {TensorSlotName::INPUT_BIAS, input_bias_initializer},
        {TensorSlotName::OUTPUT_BIAS, output_bias_initializer},
    };
  } else {
    return std::map<TensorSlotName, InitializerAttrs>{
        {TensorSlotName::WEIGHT, weights_initializer},
    };
  }
}

} // namespace FlexFlow
