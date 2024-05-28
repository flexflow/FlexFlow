#include "test/utils/doctest.h"
#include "op-attrs/ops/attention.h"
#include "utils/integer_conversions.h"
#include "op-attrs/parallel_tensor_shape.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(MultiHeadAttentionAttrs, TensorShape, TensorShape, TensorShape)") {
    int embed_dim = 32;

    /* Parameter meanings match those at 
     * https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
     */
    MultiHeadAttentionAttrs attrs = {
      /*embed_dim=*/embed_dim,
      /*num_heads=*/10,
      /*kdim=*/embed_dim,
      /*vdim=*/embed_dim,
      /*dropout=*/0.0,
      /*bias=*/true,
      /*add_bias_kv=*/false,
      /*add_zero_attn=*/false,
    };

    size_t batch_size = 40;
    size_t seq_len = 48;

    TensorShape input_q = {
      TensorDims{
        FFOrdered<size_t>{
          batch_size,
          seq_len,
          size_t_from_int(attrs.embed_dim),
        }
      },
      DataType::FLOAT,
    };

    TensorShape input_k = {
      TensorDims{
        FFOrdered<size_t>{
          batch_size,
          seq_len,
          size_t_from_int(attrs.kdim),
        },
      },
      DataType::FLOAT,
    };

    TensorShape input_v = {
      TensorDims{
        FFOrdered<size_t>{
          batch_size,
          seq_len,
          size_t_from_int(attrs.vdim),
        },
      },
      DataType::FLOAT,
    };

    SUBCASE("get_output_shape") {
      tl::expected<TensorShape, std::string> result = get_output_shape(attrs, input_q, input_k, input_v);

      tl::expected<TensorShape, std::string> correct = TensorShape{
        TensorDims{
          FFOrdered<size_t>{
            batch_size,
            seq_len, 
            size_t_from_int(attrs.embed_dim),
          }
        },
        DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("get_weights_shape") {
      tl::expected<TensorShape, std::string> result = get_weights_shape(attrs, input_q, input_k, input_v);

      int qProjPerHeadWeightSize = attrs.kdim * dim_at_idx(input_q, ff_dim_t{-1});
      int kProjPerHeadWeightSize = attrs.kdim * dim_at_idx(input_k, ff_dim_t{-1});
      int vProjPerHeadWeightSize = attrs.vdim * dim_at_idx(input_v, ff_dim_t{-1});
      int oProjPerHeadWeightSize = attrs.embed_dim * attrs.vdim;
      int perHeadWeightSize = qProjPerHeadWeightSize + kProjPerHeadWeightSize + vProjPerHeadWeightSize + oProjPerHeadWeightSize;

      tl::expected<TensorShape, std::string> correct = TensorShape{
        TensorDims{
          FFOrdered<size_t>{
            size_t_from_int(perHeadWeightSize),
            size_t_from_int(attrs.num_heads),
          }
        },
        DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("parallel shape inference for MultiHeadAttentionAttrs") {
    int embed_dim = 32;

    /* Parameter meanings can be found at 
     * https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
     */
    MultiHeadAttentionAttrs attrs = {
      /*embed_dim=*/embed_dim,
      /*num_heads=*/10,
      /*kdim=*/embed_dim,
      /*vdim=*/embed_dim,
      /*dropout=*/0.0,
      /*bias=*/true,
      /*add_bias_kv=*/false,
      /*add_zero_attn=*/false,
    };

    size_t batchsize = 40;
    size_t seq_len = 48;
    size_t q_size = 56;
    size_t k_size = 64;
    size_t v_size = 72;

    TensorShape unpar_q_shape = TensorShape{
      TensorDims{
        FFOrdered<size_t>{
          batchsize,
          seq_len,
          q_size,
        },
      },
      DataType::FLOAT, 
    };

    TensorShape unpar_k_shape = TensorShape{
      TensorDims{
        FFOrdered<size_t>{
          batchsize,
          seq_len,
          k_size,
        },
      },
      DataType::FLOAT, 
    };

    TensorShape unpar_v_shape = TensorShape{
      TensorDims{
        FFOrdered<size_t>{
          batchsize,
          seq_len,
          v_size,
        },
      },
      DataType::FLOAT, 
    };

    tl::expected<TensorShape, std::string> result_unpar_o_shape = get_output_shape(attrs, unpar_q_shape, unpar_k_shape, unpar_v_shape);
    REQUIRE(result_unpar_o_shape.has_value());
    TensorShape unpar_o_shape = result_unpar_o_shape.value();

    tl::expected<TensorShape, std::string> result_unpar_w_shape = get_weights_shape(attrs, unpar_q_shape, unpar_k_shape, unpar_v_shape);
    REQUIRE(result_unpar_o_shape.has_value());
    TensorShape unpar_w_shape = result_unpar_w_shape.value();

    auto make_q = [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o_batch, int o_seq_len, int o_q) {
      return lift_to_parallel_with_degrees(unpar_q_shape, o_sum, o_eq, FFOrdered<int>{o_batch, o_seq_len, o_q});
    };

    auto make_k = [&](int o_sum, int o_eq, int o_batch, int o_seq_len, int o_k) {
      return lift_to_parallel_with_degrees(unpar_k_shape, o_sum, o_eq, FFOrdered<int>{o_batch, o_seq_len, o_k});
    };

    auto make_v = [&](int o_sum, int o_eq, int o_batch, int o_seq_len, int o_v) {
      return lift_to_parallel_with_degrees(unpar_v_shape, o_sum, o_eq, FFOrdered<int>{o_batch, o_seq_len, o_v});
    };

    auto make_o = [&](int o_sum, int o_eq, int o_batch, int o_seq_len, int o_o) {
      return lift_to_parallel_with_degrees(unpar_o_shape, o_sum, o_eq, FFOrdered<int>{o_batch, o_seq_len, o_o});
    };

    auto make_w = [&](int o_sum, int o_eq, int o_e, int o_h) {
      return lift_to_parallel_with_degrees(unpar_w_shape, o_sum, o_eq, FFOrdered<int>{o_e, o_h});
    };

    SUBCASE("data parallelism") {
      int o_b = 4;
      ParallelTensorShape q = make_q(1, 1, o_b, 1, 1);
      ParallelTensorShape k = make_k(1, 1, o_b, 1, 1);
      ParallelTensorShape v = make_v(1, 1, o_b, 1, 1);

      tl::expected<ParallelTensorShape, std::string> result_o = get_output_shape(attrs, q, k, v);
      tl::expected<ParallelTensorShape, std::string> correct_o = make_o(1, 1, o_b, 1, 1);

      CHECK(result_o == correct_o);

      tl::expected<ParallelTensorShape, std::string> result_w = get_weights_shape(attrs, q, k, v);
      tl::expected<ParallelTensorShape, std::string> correct_w = make_w(1, o_b, 1, 1);

      CHECK(result_w == correct_w);
    }

    SUBCASE("attention head parallelism") {
      int o_h = 2;
      ParallelTensorShape q = make_q(1, o_h, 1, 1, 1);
      ParallelTensorShape k = make_k(1, o_h, 1, 1, 1);
      ParallelTensorShape v = make_v(1, o_h, 1, 1, 1);

      tl::expected<ParallelTensorShape, std::string> result_o = get_output_shape(attrs, q, k, v);
      tl::expected<ParallelTensorShape, std::string> correct_o = make_o(o_h, 1, 1, 1, 1);

      CHECK(result_o == correct_o);

      tl::expected<ParallelTensorShape, std::string> result_w = get_weights_shape(attrs, q, k, v);
      tl::expected<ParallelTensorShape, std::string> correct_w = make_w(1, 1, 1, o_h);

      CHECK(result_w == correct_w);
    }

    SUBCASE("combined data & attention head parallelism") {
      int o_b = 4;
      int o_h = 2;
      ParallelTensorShape q = make_q(1, o_h, o_b, 1, 1);
      ParallelTensorShape k = make_k(1, o_h, o_b, 1, 1);
      ParallelTensorShape v = make_v(1, o_h, o_b, 1, 1);

      tl::expected<ParallelTensorShape, std::string> result_o = get_output_shape(attrs, q, k, v);
      tl::expected<ParallelTensorShape, std::string> correct_o = make_o(o_h, 1, o_b, 1, 1);

      CHECK(result_o == correct_o);

      tl::expected<ParallelTensorShape, std::string> result_w = get_weights_shape(attrs, q, k, v);
      tl::expected<ParallelTensorShape, std::string> correct_w = make_w(1, o_b, 1, o_h);

      CHECK(result_w == correct_w);
    }
  }
}
