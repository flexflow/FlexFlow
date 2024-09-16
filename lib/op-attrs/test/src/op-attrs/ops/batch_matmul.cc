#include "op-attrs/ops/batch_matmul.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(BatchMatmulAttrs, TensorShape)") {
    size_t b = 4;
    size_t m = 6;
    size_t n = 8;
    size_t p = 10;

    BatchMatmulAttrs attrs = BatchMatmulAttrs{
        /*a_seq_length_dim=*/0, // TODO figure out if these arguments are still
                                // relevant
        /*b_seq_length_dim=*/0,
    };

    TensorShape input_lhs_shape = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                b,
                n,
                m,
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("valid") {
      TensorShape input_rhs_shape = TensorShape{
          TensorDims{
              FFOrdered<size_t>{
                  b,
                  m,
                  p,
              },
          },
          DataType::FLOAT,
      };

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_lhs_shape, input_rhs_shape);

      tl::expected<TensorShape, std::string> correct_output_shape = TensorShape{
          TensorDims{
              FFOrdered<size_t>{
                  b,
                  n,
                  p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct_output_shape);
    }

    SUBCASE("mismatched b") {
      TensorShape input_rhs_shape = TensorShape{
          TensorDims{
              FFOrdered<size_t>{
                  b + 1,
                  m,
                  p,
              },
          },
          DataType::FLOAT,
      };

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_lhs_shape, input_rhs_shape);

      CHECK(!result.has_value());
    }

    SUBCASE("mismatched m") {
      TensorShape input_rhs_shape = TensorShape{
          TensorDims{
              FFOrdered<size_t>{
                  b,
                  m + 1,
                  p,
              },
          },
          DataType::FLOAT,
      };

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_lhs_shape, input_rhs_shape);

      CHECK(!result.has_value());
    }
  }

  TEST_CASE("get_output_shape(BatchMatmulAttrs, ParallelTensorShape)") {
    size_t b = 2 * 2;
    int o_b = 2;
    size_t m = 3 * 3;
    int o_m = 3;
    size_t n = 5 * 5;
    int o_n = 5;
    size_t p = 7 * 7;
    int o_p = 7;
    int o_sum = 11;

    BatchMatmulAttrs attrs = BatchMatmulAttrs{
        /*a_seq_length_dim=*/0, // TODO figure out if these arguments are still
                                // relevant
        /*b_seq_length_dim=*/0,
    };

    auto make_lhs = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        int o_b,
                        int o_n,
                        int o_m) {
      return ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{b, o_b},
                  ShardParallelDim{n, o_n},
                  ShardParallelDim{m, o_m},
              },
              ReplicaParallelDimSet{
                  o_sum,
                  o_eq,
              },
          },
          DataType::FLOAT,
      };
    };

    auto make_rhs = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        int o_b,
                        int o_m,
                        int o_p) {
      return ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{b, o_b},
                  ShardParallelDim{m, o_m},
                  ShardParallelDim{p, o_p},
              },
              ReplicaParallelDimSet{
                  o_sum,
                  o_eq,
              },
          },
          DataType::FLOAT,
      };
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           int o_b,
                           int o_n,
                           int o_p) {
      return ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{b, o_b},
                  ShardParallelDim{n, o_n},
                  ShardParallelDim{p, o_p},
              },
              ReplicaParallelDimSet{
                  o_sum,
                  o_eq,
              },
          },
          DataType::FLOAT,
      };
    };

    SUBCASE("data parallel") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1}, DiscardCopyDegree{1}, o_b, 1, 1),
          make_rhs(SumDegree{1}, DiscardCopyDegree{1}, o_b, 1, 1));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{1}, DiscardCopyDegree{1}, o_b, 1, 1);

      CHECK(result == correct);
    }

    SUBCASE("n parallel") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1}, DiscardCopyDegree{1}, 1, o_n, 1),
          make_rhs(SumDegree{1}, DiscardCopyDegree{o_n}, 1, 1, 1));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{1}, DiscardCopyDegree{1}, 1, o_n, 1);

      CHECK(result == correct);
    }

    SUBCASE("p parallel") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1}, DiscardCopyDegree{o_p}, 1, 1, 1),
          make_rhs(SumDegree{1}, DiscardCopyDegree{1}, 1, 1, o_p));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{1}, DiscardCopyDegree{1}, 1, 1, o_p);

      CHECK(result == correct);
    }

    SUBCASE("reduction parallel") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1}, DiscardCopyDegree{1}, 1, 1, o_m),
          make_rhs(SumDegree{1}, DiscardCopyDegree{1}, 1, o_m, 1));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_m}, DiscardCopyDegree{1}, 1, 1, 1);

      CHECK(result == correct);
    }

    SUBCASE("propagate reduction lhs") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{1}, 1, 1, 1),
          make_rhs(SumDegree{1}, DiscardCopyDegree{o_sum}, 1, 1, 1));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum}, DiscardCopyDegree{1}, 1, 1, 1);

      CHECK(result == correct);
    }

    SUBCASE("propagate reduction rhs") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1}, DiscardCopyDegree{o_sum}, 1, 1, 1),
          make_rhs(SumDegree{o_sum}, DiscardCopyDegree{1}, 1, 1, 1));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum}, DiscardCopyDegree{1}, 1, 1, 1);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & reduction rhs") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1, 1, 1),
          make_rhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1, 1, 1));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum * o_sum}, DiscardCopyDegree{1}, 1, 1, 1);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & rhs (invalid)") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{1}, 1, 1, 1),
          make_rhs(SumDegree{o_sum}, DiscardCopyDegree{1}, 1, 1, 1));

      CHECK_MESSAGE(
          !result.has_value(), "Unexpected successful value: ", result);
    }

    SUBCASE("reduction lhs & n") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{1}, 1, o_n, 1),
          make_rhs(SumDegree{1}, DiscardCopyDegree{o_sum * o_n}, 1, 1, 1));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum}, DiscardCopyDegree{1}, 1, o_n, 1);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & reduction rhs & n") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1, o_n, 1),
          make_rhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum * o_n}, 1, 1, 1));
      tl::expected<ParallelTensorShape, std::string> correct = make_output(
          SumDegree{o_sum * o_sum}, DiscardCopyDegree{1}, 1, o_n, 1);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & reduction rhs & n & m") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1, o_n, o_m),
          make_rhs(
              SumDegree{o_sum}, DiscardCopyDegree{o_sum * o_n}, 1, o_m, 1));
      tl::expected<ParallelTensorShape, std::string> correct = make_output(
          SumDegree{o_sum * o_sum * o_m}, DiscardCopyDegree{1}, 1, o_n, 1);

      CHECK(result == correct);
    }
  }
}
