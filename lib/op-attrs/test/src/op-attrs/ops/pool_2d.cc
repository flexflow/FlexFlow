#include "op-attrs/ops/pool_2d.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("make_adaptive_pool2d") {
    size_t input_n = 10;
    size_t input_c = 11;
    size_t input_h = 15;
    size_t input_w = 20;
    Activation activation = Activation::RELU;
    PoolOp op = PoolOp::AVG;

    TensorDims input_dims =
        TensorDims{FFOrdered<size_t>{input_n, input_c, input_h, input_w}};

    SUBCASE("input_h divisible by output_h && input_w divisible by output_w") {
      int output_h = 5;
      int output_w = 2;

      Pool2DAttrs correct_attrs = Pool2DAttrs{
          /*kernel_h=*/3,
          /*kernel_w=*/10,
          /*stride_h=*/3,
          /*stride_w=*/10,
          /*padding_h=*/0,
          /*padding_w=*/0,
          /*pool_type=*/op,
          /*activation=*/activation,
      };

      SUBCASE("returns correct attrs") {
        tl::expected<Pool2DAttrs, std::string> result =
            make_adaptive_pool2d_attrs(
                input_dims, output_h, output_w, op, activation);
        tl::expected<Pool2DAttrs, std::string> correct = correct_attrs;

        CHECK(result == correct);
      }

      SUBCASE(
          "confirm that output shape is as expected for the expected attrs") {
        TensorShape input_shape = TensorShape{input_dims, DataType::FLOAT};

        tl::expected<TensorShape, std::string> result =
            get_output_shape(correct_attrs, input_shape);
        tl::expected<TensorShape, std::string> correct = TensorShape{
            TensorDims{FFOrdered<size_t>{
                input_n,
                input_c,
                size_t_from_int(output_h),
                size_t_from_int(output_w),
            }},
            DataType::FLOAT,
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("input_h not divisible by output_h") {
      int output_h = 6;
      int output_w = 2;

      std::optional<Pool2DAttrs> result =
          optional_from_expected(make_adaptive_pool2d_attrs(
              input_dims, output_h, output_w, op, activation));
      std::optional<Pool2DAttrs> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input_w not divisible by output_w") {
      int output_h = 5;
      int output_w = 3;

      std::optional<Pool2DAttrs> result =
          optional_from_expected(make_adaptive_pool2d_attrs(
              input_dims, output_h, output_w, op, activation));
      std::optional<Pool2DAttrs> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input_h == output_h and input_w == output_w") {
      int output_h = input_h;
      int output_w = input_w;

      Pool2DAttrs correct_attrs = Pool2DAttrs{
          /*kernel_h=*/1,
          /*kernel_w=*/1,
          /*stride_h=*/1,
          /*stride_w=*/1,
          /*padding_h=*/0,
          /*padding_w=*/0,
          /*pool_type=*/op,
          /*activation=*/activation,
      };

      SUBCASE("returns correct attrs") {
        tl::expected<Pool2DAttrs, std::string> result =
            make_adaptive_pool2d_attrs(
                input_dims, output_h, output_w, op, activation);
        tl::expected<Pool2DAttrs, std::string> correct = correct_attrs;

        CHECK(result == correct);
      }

      SUBCASE(
          "confirm that output shape is as expected for the expected attrs") {
        TensorShape input_shape = TensorShape{input_dims, DataType::FLOAT};

        tl::expected<TensorShape, std::string> result =
            get_output_shape(correct_attrs, input_shape);
        tl::expected<TensorShape, std::string> correct = input_shape;

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("get_output_shape(Pool2DAttrs, TensorShape)") {
    Pool2DAttrs attrs = Pool2DAttrs{
        /*kernel_h=*/3,
        /*kernel_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        /*padding_h=*/1,
        /*padding_w=*/1,
        /*pool_type=*/PoolOp::MAX,
        /*activation=*/std::nullopt,
    };

    SUBCASE("fails on non-4d inputs") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<size_t>{
              10,
              12,
              14,
          }},
          DataType::FLOAT,
      };

      std::optional<TensorShape> result =
          optional_from_expected(get_output_shape(attrs, input));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("4d input") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<size_t>{11, 13, 12, 6}},
          DataType::FLOAT,
      };

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input);
      tl::expected<TensorShape, std::string> correct = TensorShape{
          TensorDims{FFOrdered<size_t>{11, 13, 6, 4}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_output_parallel_dim_degrees(Pool2DAttrs, "
            "ParallelTensorDimDegrees)") {
    auto make_attrs = [](PoolOp pool_type,
                         std::optional<Activation> const &activation) {
      return Pool2DAttrs{
          /*kernel_h=*/3,
          /*kernel_w=*/2,
          /*stride_h=*/2,
          /*stride_w=*/2,
          /*padding_h=*/1,
          /*padding_w=*/1,
          /*pool_type=*/pool_type,
          /*activation=*/activation,
      };
    };

    SUBCASE("allows data parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1},
          DiscardCopyDegree{1},
          FFOrdered<int>{
              4,
              1,
              1,
              1,
          },
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result =
          get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("allows arbitrary input sharding parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1},
          DiscardCopyDegree{1},
          FFOrdered<int>{
              4,
              2,
              5,
              6,
          },
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result =
          get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("allows discard copy parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1},
          DiscardCopyDegree{3},
          FFOrdered<int>{
              1,
              1,
              1,
              1,
          },
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result =
          get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("sum parallelism") {
      SUBCASE("without activation") {
        SUBCASE("PoolOp::MAX does not allow sum parallelism") {
          Pool2DAttrs attrs =
              make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

          ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
              SumDegree{2},
              DiscardCopyDegree{1},
              FFOrdered<int>{
                  1,
                  1,
                  1,
                  1,
              },
          };

          std::optional<ParallelTensorDimDegrees> result =
              optional_from_expected(
                  get_output_parallel_dim_degrees(attrs, input));
          std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

          CHECK(result == correct);
        }

        SUBCASE("PoolOp::AVG does allow sum parallelism") {
          Pool2DAttrs attrs =
              make_attrs(PoolOp::AVG, /*activation=*/std::nullopt);

          ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
              SumDegree{2},
              DiscardCopyDegree{1},
              FFOrdered<int>{
                  1,
                  1,
                  1,
                  1,
              },
          };

          tl::expected<ParallelTensorDimDegrees, std::string> result =
              get_output_parallel_dim_degrees(attrs, input);
          tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

          CHECK(result == correct);
        }
      }

      SUBCASE("with activation does not allow sum parallelism") {
        Pool2DAttrs attrs =
            make_attrs(PoolOp::AVG, /*activation=*/Activation::RELU);

        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{2},
            DiscardCopyDegree{1},
            FFOrdered<int>{
                1,
                1,
                1,
                1,
            },
        };

        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
            get_output_parallel_dim_degrees(attrs, input));
        std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("get_output_shape(Pool2DAttrs, ParallelTensorShape)") {
    // this function is mostly covered by the tests above, so we
    // just do a single test to make sure it works/exists

    Pool2DAttrs attrs = Pool2DAttrs{
        /*kernel_h=*/3,
        /*kernel_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        /*padding_h=*/1,
        /*padding_w=*/1,
        /*pool_type=*/PoolOp::MAX,
        /*activation=*/std::nullopt,
    };

    SUBCASE("valid parallelism") {
      ParallelTensorShape input = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14, 7},
                  ShardParallelDim{16, 8},
                  ShardParallelDim{12, 3},
                  ShardParallelDim{6, 2},
              },
              ReplicaParallelDimSet{
                  SumDegree{1},
                  DiscardCopyDegree{2},
              },
          },
          DataType::FLOAT,
      };

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input);
      tl::expected<ParallelTensorShape, std::string> correct =
          ParallelTensorShape{
              ParallelTensorDims{
                  FFOrdered<ShardParallelDim>{
                      ShardParallelDim{14, 7},
                      ShardParallelDim{16, 8},
                      ShardParallelDim{6, 3},
                      ShardParallelDim{4, 2},
                  },
                  ReplicaParallelDimSet{
                      SumDegree{1},
                      DiscardCopyDegree{2},
                  },
              },
              DataType::FLOAT,
          };
    }

    SUBCASE("invalid parallelism") {
      ParallelTensorShape input = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14, 1},
                  ShardParallelDim{16, 1},
                  ShardParallelDim{12, 1},
                  ShardParallelDim{6, 1},
              },
              ReplicaParallelDimSet{
                  SumDegree{2},
                  DiscardCopyDegree{1},
              },
          },
          DataType::FLOAT,
      };

      std::optional<ParallelTensorShape> result =
          optional_from_expected(get_output_shape(attrs, input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
