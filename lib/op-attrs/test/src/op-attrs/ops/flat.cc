#include "op-attrs/ops/flat.h"
#include "utils/expected.h"
#include <doctest/doctest.h>
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(FlatAttrs, TensorShape)") {
    TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
        2,
        4,
        2,
        3,
      }},
      DataType::FLOAT,
    };

    SUBCASE("flatten all dims") {
      FlatAttrs attrs = FlatAttrs{
        /*start_dim=*/ff_dim_t{0},
        /*end_dim=*/ff_dim_t{4},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
        TensorDims{FFOrdered<size_t>{
          2 * 4 * 2 * 3,
        }},
        DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten trailing dims") {
      FlatAttrs attrs = FlatAttrs{
        /*start_dim=*/ff_dim_t{2},
        /*end_dim=*/ff_dim_t{4},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
        TensorDims{FFOrdered<size_t>{
          2,
          4,
          2 * 3,
        }},
        DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten leading dims") {
      FlatAttrs attrs = FlatAttrs{
        /*start_dim=*/ff_dim_t{0},
        /*end_dim=*/ff_dim_t{2},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
        TensorDims{FFOrdered<size_t>{
          2 * 4,
          2,
          3,
        }},
        DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten middle dims") {
      FlatAttrs attrs = FlatAttrs{
        /*start_dim=*/ff_dim_t{1},
        /*end_dim=*/ff_dim_t{3},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
        TensorDims{FFOrdered<size_t>{
          2,
          4 * 2,
          3,
        }},
        DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten no dims (start_dim == end_dim)") {
      FlatAttrs attrs = FlatAttrs{
        /*start_dim=*/ff_dim_t{2},
        /*end_dim=*/ff_dim_t{2},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = input_shape;

      CHECK(result == correct);
    }

    SUBCASE("flatten no dims (start_dim < end_dim)") {
      FlatAttrs attrs = FlatAttrs{
        /*start_dim=*/ff_dim_t{2},
        /*end_dim=*/ff_dim_t{1},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = input_shape;

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_output_parallel_dim_degrees(FlatAttrs, ParallelTensorDimDegrees)") {
    FlatAttrs attrs = FlatAttrs{
      /*start_dim=*/ff_dim_t{1},
      /*end_dim=*/ff_dim_t{3}
    };

    SUBCASE("allows shard parallelism in non-flattened dims") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{1},
        DiscardCopyDegree{1},
        FFOrdered<int>{2, 1, 1, 3},
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result = get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct = ParallelTensorDimDegrees{
        SumDegree{1},
        DiscardCopyDegree{1},
        FFOrdered<int>{2, 1, 3},
      };

      CHECK(result == correct);
    }

    SUBCASE("does not allow shard parallelism in flattened dims") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{1},
        DiscardCopyDegree{1},
        FFOrdered<int>{1, 1, 2, 1},
      };

      std::optional<ParallelTensorDimDegrees> result = optional_from_expected(get_output_parallel_dim_degrees(attrs, input));
      std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("allows sum parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{2},
        DiscardCopyDegree{1},
        FFOrdered<int>{1, 1, 1, 1},
      };

      std::optional<ParallelTensorDimDegrees> result = optional_from_expected(get_output_parallel_dim_degrees(attrs, input));
      std::optional<ParallelTensorDimDegrees> correct = ParallelTensorDimDegrees{
        SumDegree{2},
        DiscardCopyDegree{1},
        FFOrdered<int>{1, 1, 1},
      };

      CHECK(result == correct);
    }

    SUBCASE("allows discard copy parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{1},
        DiscardCopyDegree{2},
        FFOrdered<int>{1, 1, 1, 1},
      };

      std::optional<ParallelTensorDimDegrees> result = optional_from_expected(get_output_parallel_dim_degrees(attrs, input));
      std::optional<ParallelTensorDimDegrees> correct = ParallelTensorDimDegrees{
        SumDegree{1},
        DiscardCopyDegree{2},
        FFOrdered<int>{1, 1, 1},
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_output_shape(FlatAttrs, ParallelTensorShape)") {
    // since most of the edge cases are already tested in get_output_shape(FlatAttrs, TensorShape) 
    // and get_output_parallel_dim_degrees(FlatAttrs, ParallelTensorDimDegrees), here we just do 
    // a basic check that they compose
   
    ParallelTensorShape input_shape = ParallelTensorShape{
      ParallelTensorDims{
        FFOrdered<ShardParallelDim>{
          ShardParallelDim{4, 2},
          ShardParallelDim{8, 1},
          ShardParallelDim{6, 1},
          ShardParallelDim{9, 3},
        },
        ReplicaParallelDimSet{
          SumDegree{7},
          DiscardCopyDegree{5},
        },
      },
      DataType::FLOAT,
    };

    FlatAttrs attrs = FlatAttrs{
      /*start_dim=*/ff_dim_t{1},
      /*end_dim=*/ff_dim_t{3},
    };

    tl::expected<ParallelTensorShape, std::string> result = get_output_shape(attrs, input_shape);
    tl::expected<ParallelTensorShape, std::string> correct = ParallelTensorShape{
      ParallelTensorDims{
        FFOrdered<ShardParallelDim>{
          ShardParallelDim{4, 2},
          ShardParallelDim{8*6, 1},
          ShardParallelDim{9, 3},
        },
        ReplicaParallelDimSet{
          SumDegree{7},
          DiscardCopyDegree{5},
        },
      },
      DataType::FLOAT,
    };

    CHECK(result == correct);
  }
}
