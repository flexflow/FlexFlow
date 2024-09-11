#include <doctest/doctest.h>
#include "op-attrs/ops/pool_2d.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"

using namespace ::FlexFlow; 

TEST_SUITE(FF_TEST_SUITE) {
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
          10, 12, 14,
        }},
        DataType::FLOAT,
      };

      std::optional<TensorShape> result = optional_from_expected(get_output_shape(attrs, input));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("4d input") {
      TensorShape input = TensorShape{
        TensorDims{FFOrdered<size_t>{
          11, 13, 12, 6
        }},
        DataType::FLOAT,
      };

      tl::expected<TensorShape, std::string> result = get_output_shape(attrs, input);
      tl::expected<TensorShape, std::string> correct = TensorShape{
        TensorDims{FFOrdered<size_t>{
          11, 13, 6, 4
        }},
        DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_output_parallel_dim_degrees(Pool2DAttrs, ParallelTensorDimDegrees)") {
    auto make_attrs = [](PoolOp pool_type, std::optional<Activation> const &activation) {
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
          4, 1, 1, 1,
        },
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result = get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("allows arbitrary input sharding parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{1},
        DiscardCopyDegree{1},
        FFOrdered<int>{
          4, 2, 5, 6,
        },
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result = get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("allows discard copy parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{1},
        DiscardCopyDegree{3},
        FFOrdered<int>{
          1, 1, 1, 1,
        },
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result = get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("sum parallelism") {
      SUBCASE("without activation") {
        SUBCASE("PoolOp::MAX does not allow sum parallelism") {
          Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

          ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{2},
            DiscardCopyDegree{1},
            FFOrdered<int>{
              1, 1, 1, 1,
            },
          };

          std::optional<ParallelTensorDimDegrees> result = optional_from_expected(get_output_parallel_dim_degrees(attrs, input));
          std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

          CHECK(result == correct);
        }

        SUBCASE("PoolOp::AVG does allow sum parallelism") {
          Pool2DAttrs attrs = make_attrs(PoolOp::AVG, /*activation=*/std::nullopt);

          ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{2},
            DiscardCopyDegree{1},
            FFOrdered<int>{
              1, 1, 1, 1,
            },
          };

          tl::expected<ParallelTensorDimDegrees, std::string> result = get_output_parallel_dim_degrees(attrs, input);
          tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

          CHECK(result == correct);
        }
      }
      
      SUBCASE("with activation does not allow sum parallelism") {
        Pool2DAttrs attrs = make_attrs(PoolOp::AVG, /*activation=*/Activation::RELU);

        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{2},
          DiscardCopyDegree{1},
          FFOrdered<int>{
            1, 1, 1, 1,
          },
        };

        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(get_output_parallel_dim_degrees(attrs, input));
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
      
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(attrs, input);
      tl::expected<ParallelTensorShape, std::string> correct = ParallelTensorShape{
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

      std::optional<ParallelTensorShape> result = optional_from_expected(get_output_shape(attrs, input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
