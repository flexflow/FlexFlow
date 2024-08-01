#include "op-attrs/ops/attention.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest.h"
#include "utils/integer_conversions.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(MultiHeadAttentionAttrs, TensorShape, "
            "TensorShape, TensorShape)") {
    int embed_dim = 32;
    int num_heads = 10;

    /* Parameter meanings match those at
     * https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
     */
    MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/embed_dim,
        /*num_heads=*/num_heads,
        /*kdim=*/embed_dim,
        /*vdim=*/embed_dim,
        /*dropout=*/0.0,
        /*bias=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    size_t batch_size = 40;
    size_t seq_len = 48;
    size_t feature_size = 36;

    TensorShape input_q = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                batch_size,
                seq_len,
                feature_size,
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_k = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                batch_size,
                seq_len,
                feature_size,
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_v = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                batch_size,
                seq_len,
                feature_size,
            },
        },
        DataType::FLOAT,
    };

    TensorShape output = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                batch_size,
                seq_len,
                size_t_from_int(attrs.embed_dim),
            },
        },
        DataType::FLOAT,
    };

    TensorShape weights = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                (feature_size * embed_dim) * 3 + (embed_dim * embed_dim),
                size_t_from_int(num_heads),
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_bias = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                size_t_from_int(embed_dim * 3),
            },
        },
        DataType::FLOAT,
    };

    TensorShape output_bias = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                size_t_from_int(embed_dim),
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("get_output_shape") {
      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_q, input_k, input_v);

      tl::expected<TensorShape, std::string> correct = output;
      CHECK(result == correct);
    }

    SUBCASE("get_weights_shape") {
      tl::expected<TensorShape, std::string> result =
          get_weights_shape(attrs, input_q, input_k, input_v);

      tl::expected<TensorShape, std::string> correct = weights;
      CHECK(result == correct);
    }

    SUBCASE("get_input_bias_shape") {
      tl::expected<TensorShape, std::string> result =
          get_input_bias_shape(attrs, input_q, input_k, input_v);
      tl::expected<TensorShape, std::string> correct = input_bias;
      CHECK(result == correct);
    }

    SUBCASE("get_output_bias_shape") {
      tl::expected<TensorShape, std::string> result =
          get_output_bias_shape(attrs, input_q, input_k, input_v);
      tl::expected<TensorShape, std::string> correct = output_bias;
      CHECK(result == correct);
    }

    SUBCASE("parallel shape inference") {
      auto make_q = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        int o_batch,
                        int o_seq_len,
                        int o_q) {
        return lift_to_parallel_with_degrees(
            input_q, o_sum, o_eq, FFOrdered<int>{o_batch, o_seq_len, o_q});
      };

      auto make_k = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        int o_batch,
                        int o_seq_len,
                        int o_k) {
        return lift_to_parallel_with_degrees(
            input_k, o_sum, o_eq, FFOrdered<int>{o_batch, o_seq_len, o_k});
      };

      auto make_v = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        int o_batch,
                        int o_seq_len,
                        int o_v) {
        return lift_to_parallel_with_degrees(
            input_v, o_sum, o_eq, FFOrdered<int>{o_batch, o_seq_len, o_v});
      };

      auto make_o = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        int o_batch,
                        int o_seq_len,
                        int o_o) {
        return lift_to_parallel_with_degrees(
            output, o_sum, o_eq, FFOrdered<int>{o_batch, o_seq_len, o_o});
      };

      auto make_w =
          [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o_e, int o_h) {
            return lift_to_parallel_with_degrees(
                weights, o_sum, o_eq, FFOrdered<int>{o_e, o_h});
          };

      auto make_input_bias =
          [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o_in_proj_channel) {
            return lift_to_parallel_with_degrees(
                input_bias, o_sum, o_eq, FFOrdered<int>{o_in_proj_channel});
          };

      auto make_output_bias =
          [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o_out_proj_channel) {
            return lift_to_parallel_with_degrees(
                output_bias, o_sum, o_eq, FFOrdered<int>{o_out_proj_channel});
          };

      SUBCASE("data parallelism") {
        int o_b = 4;
        ParallelTensorShape q =
            make_q(SumDegree{1}, DiscardCopyDegree{1}, o_b, 1, 1);
        ParallelTensorShape k =
            make_k(SumDegree{1}, DiscardCopyDegree{1}, o_b, 1, 1);
        ParallelTensorShape v =
            make_v(SumDegree{1}, DiscardCopyDegree{1}, o_b, 1, 1);

        SUBCASE("get_output_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_o(SumDegree{1}, DiscardCopyDegree{1}, o_b, 1, 1);
          CHECK(result == correct);
        }

        SUBCASE("get_weights_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_weights_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_w(SumDegree{1}, DiscardCopyDegree{o_b}, 1, 1);
          CHECK(result == correct);
        }

        SUBCASE("get_input_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_input_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_input_bias(SumDegree{1}, DiscardCopyDegree{o_b}, 1);
          CHECK(result == correct);
        }

        SUBCASE("get_output_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_output_bias(SumDegree{1}, DiscardCopyDegree{o_b}, 1);
          CHECK(result == correct);
        }
      }

      SUBCASE("attention head parallelism") {
        int o_h = 2;
        ParallelTensorShape q =
            make_q(SumDegree{1}, DiscardCopyDegree{o_h}, 1, 1, 1);
        ParallelTensorShape k =
            make_k(SumDegree{1}, DiscardCopyDegree{o_h}, 1, 1, 1);
        ParallelTensorShape v =
            make_v(SumDegree{1}, DiscardCopyDegree{o_h}, 1, 1, 1);

        SUBCASE("get_output_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_o(SumDegree{o_h}, DiscardCopyDegree{1}, 1, 1, 1);
          CHECK(result == correct);
        }

        SUBCASE("get_weight_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_weights_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_w(SumDegree{1}, DiscardCopyDegree{1}, 1, o_h);
          CHECK(result == correct);
        }

        SUBCASE("get_input_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_input_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_input_bias(SumDegree{1}, DiscardCopyDegree{o_h}, 1);
          CHECK(result == correct);
        }

        SUBCASE("get_output_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_output_bias(SumDegree{1}, DiscardCopyDegree{o_h}, 1);
          CHECK(result == correct);
        }
      }

      SUBCASE("combined data & attention head parallelism") {
        int o_b = 4;
        int o_h = 2;
        ParallelTensorShape q =
            make_q(SumDegree{1}, DiscardCopyDegree{o_h}, o_b, 1, 1);
        ParallelTensorShape k =
            make_k(SumDegree{1}, DiscardCopyDegree{o_h}, o_b, 1, 1);
        ParallelTensorShape v =
            make_v(SumDegree{1}, DiscardCopyDegree{o_h}, o_b, 1, 1);

        SUBCASE("get_output_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_o(SumDegree{o_h}, DiscardCopyDegree{1}, o_b, 1, 1);
          CHECK(result == correct);
        }

        SUBCASE("get_weights_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_weights_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_w(SumDegree{1}, DiscardCopyDegree{o_b}, 1, o_h);
          CHECK(result == correct);
        }

        SUBCASE("get_input_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_input_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_input_bias(SumDegree{1}, DiscardCopyDegree{o_b * o_h}, 1);
          CHECK(result == correct);
        }

        SUBCASE("get_output_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_output_bias(SumDegree{1}, DiscardCopyDegree{o_b * o_h}, 1);
          CHECK(result == correct);
        }
      }
    }
  }
}
