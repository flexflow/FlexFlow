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
          /*activation=*/std::nullopt,
          /*affine=*/affine,
          /*eps=*/1.0,
          /*momentum=*/0.1,
      };
    };

    SUBCASE("affine = true") {
      BatchNormAttrs attrs = make_attrs(/*affine=*/true);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_batch_norm_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
          {
              TensorSlotName::GAMMA,
              IncomingTensorRole::WEIGHT,
          },
          {
              TensorSlotName::BETA,
              IncomingTensorRole::WEIGHT,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("affine = false") {
      BatchNormAttrs attrs = make_attrs(/*affine=*/false);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_batch_norm_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("shape inference (BatchNorm)") {
    BatchNormAttrs attrs_affine_true = BatchNormAttrs{
        /*activation=*/std::nullopt,
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
        TensorDims{FFOrdered{
            12_p,
            14_p,
            16_p,
            18_p,
        }},
        DataType::FLOAT,
    };

    TensorShape output = input;

    TensorShape gamma = TensorShape{
        TensorDims{FFOrdered{
            14_p,
        }},
        DataType::FLOAT,
    };

    TensorShape beta = gamma;

    SUBCASE("get_output_shape(BatchNormAttrs, TensorShape)") {
      TensorShape result = get_output_shape(attrs_affine_true, input);
      TensorShape correct = output;

      CHECK(result == correct);
    }

    SUBCASE("get_gamma_weights_shape(BatchNormAttrs, TensorShape)") {
      SUBCASE("affine = true") {
        TensorShape result = get_gamma_weights_shape(attrs_affine_true, input);
        TensorShape correct = gamma;

        CHECK(result == correct);
      }

      SUBCASE("affine = false") {
        CHECK_THROWS(get_gamma_weights_shape(attrs_affine_false, input));
      }
    }

    SUBCASE("get_beta_weights_shape(BatchNormAttrs, TensorShape)") {
      SUBCASE("affine = true") {
        TensorShape result = get_beta_weights_shape(attrs_affine_true, input);
        TensorShape correct = beta;

        CHECK(result == correct);
      }

      SUBCASE("affine = false") {
        CHECK_THROWS(get_beta_weights_shape(attrs_affine_false, input));
      }
    }
  }

  TEST_CASE("parallel dim degree inference (BatchNormAttrs)") {
    BatchNormAttrs attrs_affine_true = BatchNormAttrs{
        /*activation=*/std::nullopt,
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
      positive_int degree = 2_p;

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{
              1_p,
              degree,
              1_p,
              1_p,
          },
      };

      SUBCASE("get_output_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        ParallelTensorDimDegrees result =
            get_output_parallel_dim_degrees(attrs_affine_true, input);
        ParallelTensorDimDegrees correct = input;

        CHECK(result == correct);
      }

      SUBCASE("get_gamma_weights_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        SUBCASE("affine = true") {
          ParallelTensorDimDegrees result =
              get_gamma_weights_parallel_dim_degrees(attrs_affine_true, input);
          ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
              SumDegree{1_p},
              DiscardCopyDegree{1_p},
              FFOrdered{degree},
          };

          CHECK(result == correct);
        }

        SUBCASE("affine = false") {
          CHECK_THROWS(get_gamma_weights_parallel_dim_degrees(
              attrs_affine_false, input));
        }
      }

      SUBCASE("get_beta_weights_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        SUBCASE("affine = true") {
          ParallelTensorDimDegrees result =
              get_beta_weights_parallel_dim_degrees(attrs_affine_true, input);
          ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
              SumDegree{1_p},
              DiscardCopyDegree{1_p},
              FFOrdered{degree},
          };

          CHECK(result == correct);
        }

        SUBCASE("affine = false") {
          CHECK_THROWS(
              get_beta_weights_parallel_dim_degrees(attrs_affine_false, input));
        }
      }
    }

    SUBCASE("partition parallelism (not in channel dim)") {
      positive_int degree = 2_p;

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 1_p, degree, 1_p},
      };

      SUBCASE("get_output_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        CHECK_THROWS(get_output_parallel_dim_degrees(attrs_affine_true, input));
      }

      SUBCASE("get_gamma_weights_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        CHECK_THROWS(
            get_gamma_weights_parallel_dim_degrees(attrs_affine_true, input));
      }

      SUBCASE("get_beta_weights_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        CHECK_THROWS(
            get_beta_weights_parallel_dim_degrees(attrs_affine_true, input));
      }
    }

    SUBCASE("sum parallelism") {
      SumDegree sum_degree = SumDegree{2_p};

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          sum_degree,
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 1_p, 1_p, 1_p},
      };

      SUBCASE("get_output_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        CHECK_THROWS(get_output_parallel_dim_degrees(attrs_affine_true, input));
      }

      SUBCASE("get_gamma_weights_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        CHECK_THROWS(
            get_gamma_weights_parallel_dim_degrees(attrs_affine_true, input));
      }

      SUBCASE("get_beta_weights_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        CHECK_THROWS(
            get_beta_weights_parallel_dim_degrees(attrs_affine_true, input));
      }
    }

    SUBCASE("discard copy parallelism") {
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_p};

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          discard_copy_degree,
          FFOrdered{1_p, 1_p, 1_p, 1_p},
      };

      SUBCASE("get_output_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        CHECK_THROWS(get_output_parallel_dim_degrees(attrs_affine_true, input));
      }

      SUBCASE("get_gamma_weights_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        CHECK_THROWS(
            get_gamma_weights_parallel_dim_degrees(attrs_affine_true, input));
      }

      SUBCASE("get_beta_weights_parallel_dim_degrees(BatchNormAttrs, "
              "ParallelTensorDimDegrees)") {
        CHECK_THROWS(
            get_beta_weights_parallel_dim_degrees(attrs_affine_true, input));
      }
    }
  }

  TEST_CASE("parallel shape inference (BatchNormAttrs)") {
    // since most of the edge cases are already tested in the above test cases
    // (i.e., shape inference and parallel degree inference)
    // here we just do a basic check that they compose

    BatchNormAttrs attrs = BatchNormAttrs{
        /*activation=*/Activation::RELU,
        /*affine=*/true,
        /*eps=*/1.0,
        /*momentum=*/0.1,
    };

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{12_p, 1_p},
                ShardParallelDim{14_p, 2_p},
                ShardParallelDim{16_p, 1_p},
                ShardParallelDim{18_p, 1_p},
            },
            ReplicaParallelDimSet{
                SumDegree{1_p},
                DiscardCopyDegree{1_p},
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("get_output_shape(BatchNormAttrs, ParallelTensorShape)") {
      ParallelTensorShape result = get_output_shape(attrs, input);
      ParallelTensorShape correct = input;

      CHECK(result == correct);
    }

    SUBCASE("get_gamma_weights_shape(BatchNormAttrs, ParallelTensorShape)") {
      ParallelTensorShape result = get_gamma_weights_shape(attrs, input);
      ParallelTensorShape correct = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_p, 2_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{1_p},
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("get_beta_weights_shape(BatchNormAttrs, ParallelTensorShape)") {
      ParallelTensorShape result = get_beta_weights_shape(attrs, input);
      ParallelTensorShape correct = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_p, 2_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{1_p},
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }
}
