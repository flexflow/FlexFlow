#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_batch_norm_incoming_tensor_roles(BatchNormAttrs)") {
    auto make_attrs = [](bool affine) {
      return BatchNormAttrs{
        /*relu=*/false,
        /*affine=*/affine,
        /*eps=*/1.0,
        /*momentum=*/0.1,
      };
    };

    SUBCASE("affine = true") {
      BatchNormAttrs attrs = make_attrs(/*affine=*/true);

      std::vector<IncomingTensorRole> result =
          get_batch_norm_incoming_tensor_roles(attrs);
      std::vector<IncomingTensorRole> correct = {
          IncomingTensorRole::INPUT,
          IncomingTensorRole::WEIGHT,
          IncomingTensorRole::WEIGHT,
      };

      CHECK(result == correct);
    }

    SUBCASE("affine = false") {
      BatchNormAttrs attrs = make_attrs(/*affine=*/false);

      std::vector<IncomingTensorRole> result =
          get_batch_norm_incoming_tensor_roles(attrs);
      std::vector<IncomingTensorRole> correct = {
          IncomingTensorRole::INPUT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("shape inference (BatchNorm)") {
    BatchNormAttrs attrs_affine_true = BatchNormAttrs{
      /*relu=*/false,
      /*affine=*/true,
      /*eps=*/1.0,
      /*momentum=*/0.1,
    };

    BatchNormAttrs attrs_affine_false = [&] {
      BatchNormAttrs attrs = attrs_affine_true;
      attrs.affine = false;
      return attrs;
    }();

    TensorShape input = TensorShape{
        TensorDims{FFOrdered<size_t>{
            12,
            14,
            16,
            18,
        }},
        DataType::FLOAT,
    };

    TensorShape output = input;

    TensorShape gamma = TensorShape{
        TensorDims{FFOrdered<size_t>{
            14,
        }},
        DataType::FLOAT,
    };

    TensorShape beta = gamma;

    SUBCASE("get_output_shape(BatchNormAttrs, TensorShape)") {
      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs_affine_true, input);
      tl::expected<TensorShape, std::string> correct = output;

      CHECK(result == correct);
    }

    SUBCASE("get_gamma_weights_shape(BatchNormAttrs, TensorShape)") {
      SUBCASE("affine = true") {
        tl::expected<TensorShape, std::string> result =
            get_gamma_weights_shape(attrs_affine_true, input);
        tl::expected<TensorShape, std::string> correct = gamma;

        CHECK(result == correct);
      }

      SUBCASE("affine = false") {
        std::optional<TensorShape> result = optional_from_expected(
            get_gamma_weights_shape(attrs_affine_false, input));
        std::optional<TensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("get_beta_weights_shape(BatchNormAttrs, TensorShape)") {
      SUBCASE("affine = true") {
        tl::expected<TensorShape, std::string> result =
            get_beta_weights_shape(attrs_affine_true, input);
        tl::expected<TensorShape, std::string> correct = beta;

        CHECK(result == correct);
      }

      SUBCASE("affine = false") {
        std::optional<TensorShape> result = optional_from_expected(
            get_beta_weights_shape(attrs_affine_false, input));
        std::optional<TensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("parallel dim degree inference (BatchNormAttrs)") {
    BatchNormAttrs attrs_affine_true = BatchNormAttrs{
      /*relu=*/false,
      /*affine=*/true,
      /*eps=*/1.0,
      /*momentum=*/0.1,
    };

    BatchNormAttrs attrs_affine_false = [&] {
      BatchNormAttrs attrs = attrs_affine_true;
      attrs.affine = false;
      return attrs;
    }();

    SUBCASE("partition parallelism (in channel dim)") {
      int degree = 2;

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{1}, 
        DiscardCopyDegree{1}, 
        FFOrdered<int>{
          1, degree, 1, 1,
        },
      };

      SUBCASE("get_output_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        tl::expected<ParallelTensorDimDegrees, std::string> result =
            get_output_parallel_dim_degrees(attrs_affine_true, input);
        tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

        CHECK(result == correct);
      }

      SUBCASE(
          "get_gamma_weights_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        SUBCASE("affine = true") {
          tl::expected<ParallelTensorDimDegrees, std::string> result =
              get_gamma_weights_parallel_dim_degrees(attrs_affine_true, input);
          tl::expected<ParallelTensorDimDegrees, std::string> correct = ParallelTensorDimDegrees{
            SumDegree{1},
            DiscardCopyDegree{1},
            FFOrdered<int>{degree},
          };

          CHECK(result == correct);
        }

        SUBCASE("affine = false") {
          std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
              get_gamma_weights_parallel_dim_degrees(attrs_affine_false, input));
          std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

          CHECK(result == correct);
        }
      }

      SUBCASE("get_beta_weights_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        SUBCASE("affine = true") {
          tl::expected<ParallelTensorDimDegrees, std::string> result =
              get_beta_weights_parallel_dim_degrees(attrs_affine_true, input);
          tl::expected<ParallelTensorDimDegrees, std::string> correct = ParallelTensorDimDegrees{
            SumDegree{1},
            DiscardCopyDegree{1},
            FFOrdered<int>{degree},
          };

          CHECK(result == correct);
        }

        SUBCASE("affine = false") {
          std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
              get_beta_weights_parallel_dim_degrees(attrs_affine_false, input));
          std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

          CHECK(result == correct);
        }
      }
    }

    SUBCASE("partition parallelism (not in channel dim)") {
      int degree = 2;

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{1},
        DiscardCopyDegree{1},
        FFOrdered<int>{1, 1, degree, 1},
      };

      SUBCASE("get_output_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
            get_output_parallel_dim_degrees(attrs_affine_true, input));
        std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

        CHECK(result == correct);
      }

      SUBCASE(
          "get_gamma_weights_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
            get_gamma_weights_parallel_dim_degrees(attrs_affine_true, input));
        std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

        CHECK(result == correct);
      }

      SUBCASE("get_beta_weights_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
            get_beta_weights_parallel_dim_degrees(attrs_affine_true, input));
        std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("sum parallelism") {
      SumDegree sum_degree = SumDegree{2};

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        sum_degree, 
        DiscardCopyDegree{1},
        FFOrdered<int>{1, 1, 1, 1},
      };

      SUBCASE("get_output_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
            get_output_parallel_dim_degrees(attrs_affine_true, input));
        std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

        CHECK(result == correct);
      }

      SUBCASE(
          "get_gamma_weights_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
            get_gamma_weights_parallel_dim_degrees(attrs_affine_true, input));
        std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

        CHECK(result == correct);
      }

      SUBCASE("get_beta_weights_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
            get_beta_weights_parallel_dim_degrees(attrs_affine_true, input));
        std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("discard copy parallelism") {
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2};

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{1},
        discard_copy_degree,
        FFOrdered<int>{1, 1, 1, 1},
      };

      SUBCASE("get_output_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
            get_output_parallel_dim_degrees(attrs_affine_true, input));
        std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

        CHECK(result == correct);
      }

      SUBCASE(
          "get_gamma_weights_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
            get_gamma_weights_parallel_dim_degrees(attrs_affine_true, input));
        std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

        CHECK(result == correct);
      }

      SUBCASE("get_beta_weights_parallel_dim_degrees(BatchNormAttrs, ParallelTensorDimDegrees)") {
        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
            get_beta_weights_parallel_dim_degrees(attrs_affine_true, input));
        std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("parallel shape inference (BatchNormAttrs)") {
    // since most of the edge cases are already tested in the above test cases 
    // (i.e., shape inference and parallel degree inference)
    // here we just do a basic check that they compose

    BatchNormAttrs attrs = BatchNormAttrs{
      /*relu=*/true,
      /*affine=*/true,
      /*eps=*/1.0,
      /*momentum=*/0.1,
    };

    ParallelTensorShape input = ParallelTensorShape{
      ParallelTensorDims{
        FFOrdered<ShardParallelDim>{
          ShardParallelDim{12, 1},
          ShardParallelDim{14, 2},
          ShardParallelDim{16, 1},
          ShardParallelDim{18, 1},
        },
        ReplicaParallelDimSet{
          SumDegree{1},
          DiscardCopyDegree{1},
        },
      },
      DataType::FLOAT,
    };
    
    SUBCASE("get_output_shape(BatchNormAttrs, ParallelTensorShape)") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(attrs, input);
      tl::expected<ParallelTensorShape, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("get_gamma_weights_shape(BatchNormAttrs, ParallelTensorShape)") {
      tl::expected<ParallelTensorShape, std::string> result = get_gamma_weights_shape(attrs, input);
      tl::expected<ParallelTensorShape, std::string> correct = ParallelTensorShape{
        ParallelTensorDims{
          FFOrdered<ShardParallelDim>{
            ShardParallelDim{14, 2},
          },
          ReplicaParallelDimSet{
            SumDegree{1},
            DiscardCopyDegree{1},
          },
        },
        DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("get_beta_weights_shape(BatchNormAttrs, ParallelTensorShape)") {
      tl::expected<ParallelTensorShape, std::string> result = get_beta_weights_shape(attrs, input);
      tl::expected<ParallelTensorShape, std::string> correct = ParallelTensorShape{
        ParallelTensorDims{
          FFOrdered<ShardParallelDim>{
            ShardParallelDim{14, 2},
          },
          ReplicaParallelDimSet{
            SumDegree{1},
            DiscardCopyDegree{1},
          },
        },
        DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }
}
