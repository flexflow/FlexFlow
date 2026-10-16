#include "op-attrs/ops/pool_2d.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("make_adaptive_pool2d") {
    positive_int input_n = 10_p;
    positive_int input_c = 11_p;
    positive_int input_h = 15_p;
    positive_int input_w = 20_p;
    Activation activation = Activation::RELU;
    PoolOp op = PoolOp::AVG;

    TensorDims input_dims =
        TensorDims{FFOrdered{input_n, input_c, input_h, input_w}};

    SUBCASE("input_h divisible by output_h && input_w divisible by output_w") {
      positive_int output_h = 5_p;
      positive_int output_w = 2_p;

      Pool2DAttrs correct_attrs = Pool2DAttrs{
          /*kernel_h=*/3_p,
          /*kernel_w=*/10_p,
          /*stride_h=*/3_p,
          /*stride_w=*/10_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*pool_type=*/op,
          /*activation=*/activation,
      };

      SUBCASE("returns correct attrs") {
        Pool2DAttrs result = make_adaptive_pool2d_attrs(
            input_dims, output_h, output_w, op, activation);
        Pool2DAttrs correct = correct_attrs;

        CHECK(result == correct);
      }

      SUBCASE(
          "confirm that output shape is as expected for the expected attrs") {
        TensorShape input_shape = TensorShape{input_dims, DataType::FLOAT};

        TensorShape result = get_output_shape(correct_attrs, input_shape);
        TensorShape correct = TensorShape{
            TensorDims{FFOrdered{
                input_n,
                input_c,
                output_h,
                output_w,
            }},
            DataType::FLOAT,
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("input_h not divisible by output_h") {
      positive_int output_h = 6_p;
      positive_int output_w = 2_p;

      CHECK_THROWS(make_adaptive_pool2d_attrs(
          input_dims, output_h, output_w, op, activation));
    }

    SUBCASE("input_w not divisible by output_w") {
      positive_int output_h = 5_p;
      positive_int output_w = 3_p;

      CHECK_THROWS(make_adaptive_pool2d_attrs(
          input_dims, output_h, output_w, op, activation));
    }

    SUBCASE("input_h == output_h and input_w == output_w") {
      positive_int output_h = input_h;
      positive_int output_w = input_w;

      Pool2DAttrs correct_attrs = Pool2DAttrs{
          /*kernel_h=*/1_p,
          /*kernel_w=*/1_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*pool_type=*/op,
          /*activation=*/activation,
      };

      SUBCASE("returns correct attrs") {
        Pool2DAttrs result = make_adaptive_pool2d_attrs(
            input_dims, output_h, output_w, op, activation);
        Pool2DAttrs correct = correct_attrs;

        CHECK(result == correct);
      }

      SUBCASE(
          "confirm that output shape is as expected for the expected attrs") {
        TensorShape input_shape = TensorShape{input_dims, DataType::FLOAT};

        TensorShape result = get_output_shape(correct_attrs, input_shape);
        TensorShape correct = input_shape;

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("get_output_shape(Pool2DAttrs, TensorShape)") {
    Pool2DAttrs attrs = Pool2DAttrs{
        /*kernel_h=*/3_p,
        /*kernel_w=*/2_p,
        /*stride_h=*/2_p,
        /*stride_w=*/2_p,
        /*padding_h=*/1_n,
        /*padding_w=*/1_n,
        /*pool_type=*/PoolOp::MAX,
        /*activation=*/std::nullopt,
    };

    SUBCASE("fails on 1d inputs") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered{
              10_p,
          }},
          DataType::FLOAT,
      };

      CHECK_THROWS(get_output_shape(attrs, input));
    }

    SUBCASE("2d input") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered{12_p, 6_p}},
          DataType::FLOAT,
      };

      TensorShape result = get_output_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{6_p, 4_p}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("4d input") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered{11_p, 13_p, 12_p, 6_p}},
          DataType::FLOAT,
      };

      TensorShape result = get_output_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{11_p, 13_p, 6_p, 4_p}},
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
          /*kernel_h=*/3_p,
          /*kernel_w=*/2_p,
          /*stride_h=*/2_p,
          /*stride_w=*/2_p,
          /*padding_h=*/1_n,
          /*padding_w=*/1_n,
          /*pool_type=*/pool_type,
          /*activation=*/activation,
      };
    };

    SUBCASE("allows data parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{
              4_p,
              1_p,
              1_p,
              1_p,
          },
      };

      ParallelTensorDimDegrees result =
          get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = input;

      CHECK(result == correct);
    }

    SUBCASE("allows arbitrary input sharding parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{
              4_p,
              2_p,
              5_p,
              6_p,
          },
      };

      ParallelTensorDimDegrees result =
          get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = input;

      CHECK(result == correct);
    }

    SUBCASE("allows discard copy parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{3_p},
          FFOrdered{
              1_p,
              1_p,
              1_p,
              1_p,
          },
      };

      ParallelTensorDimDegrees result =
          get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = input;

      CHECK(result == correct);
    }

    SUBCASE("sum parallelism") {
      SUBCASE("without activation") {
        SUBCASE("PoolOp::MAX does not allow sum parallelism") {
          Pool2DAttrs attrs =
              make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

          ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
              SumDegree{2_p},
              DiscardCopyDegree{1_p},
              FFOrdered{
                  1_p,
                  1_p,
                  1_p,
                  1_p,
              },
          };

          CHECK_THROWS(get_output_parallel_dim_degrees(attrs, input));
        }

        SUBCASE("PoolOp::AVG does allow sum parallelism") {
          Pool2DAttrs attrs =
              make_attrs(PoolOp::AVG, /*activation=*/std::nullopt);

          ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
              SumDegree{2_p},
              DiscardCopyDegree{1_p},
              FFOrdered{
                  1_p,
                  1_p,
                  1_p,
                  1_p,
              },
          };

          ParallelTensorDimDegrees result =
              get_output_parallel_dim_degrees(attrs, input);
          ParallelTensorDimDegrees correct = input;

          CHECK(result == correct);
        }
      }

      SUBCASE("with activation does not allow sum parallelism") {
        Pool2DAttrs attrs =
            make_attrs(PoolOp::AVG, /*activation=*/Activation::RELU);

        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{2_p},
            DiscardCopyDegree{1_p},
            FFOrdered{
                1_p,
                1_p,
                1_p,
                1_p,
            },
        };

        CHECK_THROWS(get_output_parallel_dim_degrees(attrs, input));
      }
    }
  }

  TEST_CASE("get_output_shape(Pool2DAttrs, ParallelTensorShape)") {
    // this function is mostly covered by the tests above, so we
    // just do a single test to make sure it works/exists

    Pool2DAttrs attrs = Pool2DAttrs{
        /*kernel_h=*/3_p,
        /*kernel_w=*/2_p,
        /*stride_h=*/2_p,
        /*stride_w=*/2_p,
        /*padding_h=*/1_n,
        /*padding_w=*/1_n,
        /*pool_type=*/PoolOp::MAX,
        /*activation=*/std::nullopt,
    };

    SUBCASE("valid parallelism") {
      ParallelTensorShape input = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_p, 7_p},
                  ShardParallelDim{16_p, 8_p},
                  ShardParallelDim{12_p, 3_p},
                  ShardParallelDim{6_p, 2_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{2_p},
              },
          },
          DataType::FLOAT,
      };

      ParallelTensorShape result = get_output_shape(attrs, input);
      ParallelTensorShape correct = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_p, 7_p},
                  ShardParallelDim{16_p, 8_p},
                  ShardParallelDim{6_p, 3_p},
                  ShardParallelDim{4_p, 2_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{2_p},
              },
          },
          DataType::FLOAT,
      };
    }

    SUBCASE("invalid parallelism") {
      ParallelTensorShape input = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_p, 1_p},
                  ShardParallelDim{16_p, 1_p},
                  ShardParallelDim{12_p, 1_p},
                  ShardParallelDim{6_p, 1_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{2_p},
                  DiscardCopyDegree{1_p},
              },
          },
          DataType::FLOAT,
      };

      CHECK_THROWS(get_output_shape(attrs, input));
    }
  }
}
