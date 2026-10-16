#include "op-attrs/ops/attention.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("attention_get_output_shape") {
    positive_int embed_dim = 32_p;
    positive_int num_heads = 10_p;

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

    positive_int batch_size = 40_p;
    positive_int seq_len = 48_p;
    positive_int feature_size = 36_p;

    TensorShape input_q = TensorShape{
        TensorDims{
            FFOrdered{
                batch_size,
                seq_len,
                feature_size,
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_k = TensorShape{
        TensorDims{
            FFOrdered{
                batch_size,
                seq_len,
                feature_size,
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_v = TensorShape{
        TensorDims{
            FFOrdered{
                batch_size,
                seq_len,
                feature_size,
            },
        },
        DataType::FLOAT,
    };

    TensorShape output = TensorShape{
        TensorDims{
            FFOrdered{
                batch_size,
                seq_len,
                attrs.embed_dim,
            },
        },
        DataType::FLOAT,
    };

    TensorShape weights = TensorShape{
        TensorDims{
            FFOrdered{
                (feature_size * embed_dim) * 3_p + (embed_dim * embed_dim),
                num_heads,
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_bias = TensorShape{
        TensorDims{
            FFOrdered{
                embed_dim * 3_p,
            },
        },
        DataType::FLOAT,
    };

    TensorShape output_bias = TensorShape{
        TensorDims{
            FFOrdered{
                embed_dim,
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("attention_get_output_shape") {
      TensorShape result =
          attention_get_output_shape(attrs, input_q, input_k, input_v);

      TensorShape correct = output;
      CHECK(result == correct);
    }

    SUBCASE("attention_get_weights_shape") {
      TensorShape result =
          attention_get_weights_shape(attrs, input_q, input_k, input_v);

      TensorShape correct = weights;
      CHECK(result == correct);
    }

    SUBCASE("attention_get_input_bias_shape") {
      TensorShape result =
          attention_get_input_bias_shape(attrs, input_q, input_k, input_v);
      TensorShape correct = input_bias;
      CHECK(result == correct);
    }

    SUBCASE("attention_get_output_bias_shape") {
      TensorShape result =
          attention_get_output_bias_shape(attrs, input_q, input_k, input_v);
      TensorShape correct = output_bias;
      CHECK(result == correct);
    }

    SUBCASE("parallel shape inference") {
      auto make_q = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        positive_int o_batch,
                        positive_int o_seq_len,
                        positive_int o_q) {
        return lift_shape_to_parallel_with_degrees(
            input_q, o_sum, o_eq, FFOrdered{o_batch, o_seq_len, o_q});
      };

      auto make_k = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        positive_int o_batch,
                        positive_int o_seq_len,
                        positive_int o_k) {
        return lift_shape_to_parallel_with_degrees(
            input_k, o_sum, o_eq, FFOrdered{o_batch, o_seq_len, o_k});
      };

      auto make_v = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        positive_int o_batch,
                        positive_int o_seq_len,
                        positive_int o_v) {
        return lift_shape_to_parallel_with_degrees(
            input_v, o_sum, o_eq, FFOrdered{o_batch, o_seq_len, o_v});
      };

      auto make_o = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        positive_int o_batch,
                        positive_int o_seq_len,
                        positive_int o_o) {
        return lift_shape_to_parallel_with_degrees(
            output, o_sum, o_eq, FFOrdered{o_batch, o_seq_len, o_o});
      };

      auto make_w = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        positive_int o_e,
                        positive_int o_h) {
        return lift_shape_to_parallel_with_degrees(
            weights, o_sum, o_eq, FFOrdered{o_e, o_h});
      };

      auto make_input_bias = [&](SumDegree o_sum,
                                 DiscardCopyDegree o_eq,
                                 positive_int o_in_proj_channel) {
        return lift_shape_to_parallel_with_degrees(
            input_bias, o_sum, o_eq, FFOrdered{o_in_proj_channel});
      };

      auto make_output_bias = [&](SumDegree o_sum,
                                  DiscardCopyDegree o_eq,
                                  positive_int o_out_proj_channel) {
        return lift_shape_to_parallel_with_degrees(
            output_bias, o_sum, o_eq, FFOrdered{o_out_proj_channel});
      };

      SUBCASE("data parallelism") {
        positive_int o_b = 4_p;
        ParallelTensorShape q =
            make_q(SumDegree{1_p}, DiscardCopyDegree{1_p}, o_b, 1_p, 1_p);
        ParallelTensorShape k =
            make_k(SumDegree{1_p}, DiscardCopyDegree{1_p}, o_b, 1_p, 1_p);
        ParallelTensorShape v =
            make_v(SumDegree{1_p}, DiscardCopyDegree{1_p}, o_b, 1_p, 1_p);

        SUBCASE("attention_get_output_parallel_shape") {
          ParallelTensorShape result =
              attention_get_output_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct =
              make_o(SumDegree{1_p}, DiscardCopyDegree{1_p}, o_b, 1_p, 1_p);
          CHECK(result == correct);
        }

        SUBCASE("attention_get_weights_parallel_shape") {
          ParallelTensorShape result =
              attention_get_weights_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct =
              make_w(SumDegree{1_p}, DiscardCopyDegree{o_b}, 1_p, 1_p);
          CHECK(result == correct);
        }

        SUBCASE("attention_get_input_bias_parallel_shape") {
          ParallelTensorShape result =
              attention_get_input_bias_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct =
              make_input_bias(SumDegree{1_p}, DiscardCopyDegree{o_b}, 1_p);
          CHECK(result == correct);
        }

        SUBCASE("attention_get_output_bias_parallel_shape") {
          ParallelTensorShape result =
              attention_get_output_bias_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct =
              make_output_bias(SumDegree{1_p}, DiscardCopyDegree{o_b}, 1_p);
          CHECK(result == correct);
        }
      }

      SUBCASE("attention head parallelism") {
        positive_int o_h = 2_p;
        ParallelTensorShape q =
            make_q(SumDegree{1_p}, DiscardCopyDegree{o_h}, 1_p, 1_p, 1_p);
        ParallelTensorShape k =
            make_k(SumDegree{1_p}, DiscardCopyDegree{o_h}, 1_p, 1_p, 1_p);
        ParallelTensorShape v =
            make_v(SumDegree{1_p}, DiscardCopyDegree{o_h}, 1_p, 1_p, 1_p);

        SUBCASE("attention_get_output_parallel_shape") {
          ParallelTensorShape result =
              attention_get_output_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct =
              make_o(SumDegree{o_h}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);
          CHECK(result == correct);
        }

        SUBCASE("attention_get_weight_parallel_shape") {
          ParallelTensorShape result =
              attention_get_weights_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct =
              make_w(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, o_h);
          CHECK(result == correct);
        }

        SUBCASE("attention_get_input_bias_parallel_shape") {
          ParallelTensorShape result =
              attention_get_input_bias_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct =
              make_input_bias(SumDegree{1_p}, DiscardCopyDegree{o_h}, 1_p);
          CHECK(result == correct);
        }

        SUBCASE("attention_get_output_bias_parallel_shape") {
          ParallelTensorShape result =
              attention_get_output_bias_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct =
              make_output_bias(SumDegree{1_p}, DiscardCopyDegree{o_h}, 1_p);
          CHECK(result == correct);
        }
      }

      SUBCASE("combined data & attention head parallelism") {
        positive_int o_b = 4_p;
        positive_int o_h = 2_p;
        ParallelTensorShape q =
            make_q(SumDegree{1_p}, DiscardCopyDegree{o_h}, o_b, 1_p, 1_p);
        ParallelTensorShape k =
            make_k(SumDegree{1_p}, DiscardCopyDegree{o_h}, o_b, 1_p, 1_p);
        ParallelTensorShape v =
            make_v(SumDegree{1_p}, DiscardCopyDegree{o_h}, o_b, 1_p, 1_p);

        SUBCASE("attention_get_output_parallel_shape") {
          ParallelTensorShape result =
              attention_get_output_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct =
              make_o(SumDegree{o_h}, DiscardCopyDegree{1_p}, o_b, 1_p, 1_p);
          CHECK(result == correct);
        }

        SUBCASE("attention_get_weights_parallel_shape") {
          ParallelTensorShape result =
              attention_get_weights_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct =
              make_w(SumDegree{1_p}, DiscardCopyDegree{o_b}, 1_p, o_h);
          CHECK(result == correct);
        }

        SUBCASE("attention_get_input_bias_parallel_shape") {
          ParallelTensorShape result =
              attention_get_input_bias_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct = make_input_bias(
              SumDegree{1_p}, DiscardCopyDegree{o_b * o_h}, 1_p);
          CHECK(result == correct);
        }

        SUBCASE("attention_get_output_bias_parallel_shape") {
          ParallelTensorShape result =
              attention_get_output_bias_parallel_shape(attrs, q, k, v);
          ParallelTensorShape correct = make_output_bias(
              SumDegree{1_p}, DiscardCopyDegree{o_b * o_h}, 1_p);
          CHECK(result == correct);
        }
      }
    }
  }

  TEST_CASE("get_attention_incoming_tensor_roles") {
    auto make_attrs = [](bool bias) {
      return MultiHeadAttentionAttrs{
          /*embed_dim=*/32_p,
          /*num_heads=*/10_p,
          /*kdim=*/32_p,
          /*vdim=*/32_p,
          /*dropout=*/0.0,
          /*bias=*/bias,
          /*add_bias_kv=*/false,
          /*add_zero_attn=*/false,
      };
    };

    SUBCASE("without bias") {
      MultiHeadAttentionAttrs attrs = make_attrs(/*bias=*/false);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_attention_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct =
          std::map<TensorSlotName, IncomingTensorRole>{
              {
                  TensorSlotName::KEY,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::QUERY,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::VALUE,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::WEIGHT,
                  IncomingTensorRole::WEIGHT,
              },
          };

      CHECK(result == correct);
    }

    SUBCASE("with bias") {
      MultiHeadAttentionAttrs attrs = make_attrs(/*bias=*/true);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_attention_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct =
          std::map<TensorSlotName, IncomingTensorRole>{
              {
                  TensorSlotName::KEY,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::QUERY,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::VALUE,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::WEIGHT,
                  IncomingTensorRole::WEIGHT,
              },
              {
                  TensorSlotName::INPUT_BIAS,
                  IncomingTensorRole::WEIGHT,
              },
              {
                  TensorSlotName::OUTPUT_BIAS,
                  IncomingTensorRole::WEIGHT,
              },
          };

      CHECK(result == correct);
    }
  }
}
