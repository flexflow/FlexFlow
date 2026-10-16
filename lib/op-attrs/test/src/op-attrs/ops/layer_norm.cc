#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_layer_norm_incoming_tensor_roles") {
    auto make_attrs = [](bool elementwise_affine) {
      return LayerNormAttrs{
          /*axes=*/{ff_dim_t{0_n}, ff_dim_t{2_n}},
          elementwise_affine,
          /*eps=*/1.0,
      };
    };

    SUBCASE("elementwise_affine = true") {
      LayerNormAttrs attrs = make_attrs(/*elementwise_affine=*/true);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_layer_norm_incoming_tensor_roles(attrs);
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

    SUBCASE("elementwise_affine = false") {
      LayerNormAttrs attrs = make_attrs(/*elementwise_affine=*/false);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_layer_norm_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("shape inference (LayerNorm)") {
    LayerNormAttrs attrs_affine_true = LayerNormAttrs{
        /*axes=*/{ff_dim_t{1_n}, ff_dim_t{3_n}},
        /*elementwise_affine=*/true,
        /*eps=*/0.1,
    };

    LayerNormAttrs attrs_affine_false = [&] {
      LayerNormAttrs attrs = attrs_affine_true;
      attrs.elementwise_affine = false;
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
            12_p,
            16_p,
        }},
        DataType::FLOAT,
    };

    TensorShape beta = gamma;

    SUBCASE("layer_norm_get_output_shape") {
      TensorShape result =
          layer_norm_get_output_shape(attrs_affine_true, input);
      TensorShape correct = output;

      CHECK(result == correct);
    }

    SUBCASE("layer_norm_get_gamma_weights_shape") {
      SUBCASE("elementwise_affine = true") {
        TensorShape result =
            layer_norm_get_gamma_weights_shape(attrs_affine_true, input);
        TensorShape correct = gamma;

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(
            layer_norm_get_gamma_weights_shape(attrs_affine_false, input));
      }
    }

    SUBCASE("layer_norm_get_beta_weights_shape") {
      SUBCASE("elementwise_affine = true") {
        TensorShape result =
            layer_norm_get_beta_weights_shape(attrs_affine_true, input);
        TensorShape correct = beta;

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(
            layer_norm_get_beta_weights_shape(attrs_affine_false, input));
      }
    }

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          positive_int o0,
                          positive_int o1,
                          positive_int o2,
                          positive_int o3) {
      return lift_shape_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered{o0, o1, o2, o3});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o0,
                           positive_int o1,
                           positive_int o2,
                           positive_int o3) {
      return lift_shape_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered{o0, o1, o2, o3});
    };

    auto make_gamma_weights = [&](SumDegree o_sum,
                                  DiscardCopyDegree o_eq,
                                  positive_int o0,
                                  positive_int o2) {
      return lift_shape_to_parallel_with_degrees(
          gamma, o_sum, o_eq, FFOrdered{o0, o2});
    };

    auto make_beta_weights = [&](SumDegree o_sum,
                                 DiscardCopyDegree o_eq,
                                 positive_int o0,
                                 positive_int o2) {
      return lift_shape_to_parallel_with_degrees(
          beta, o_sum, o_eq, FFOrdered{o0, o2});
    };

    SUBCASE("parallel shape inference (LayerNorm)") {
      SUBCASE("partition parallelism (not in axes)") {
        positive_int degree0 = 2_p;
        positive_int degree2 = 3_p;

        ParallelTensorShape par_input = make_input(
            SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2, 1_p);

        SUBCASE("layer_norm_get_output_parallel_shape") {
          ParallelTensorShape result = layer_norm_get_output_parallel_shape(
              attrs_affine_true, par_input);
          ParallelTensorShape correct = make_output(SumDegree{1_p},
                                                    DiscardCopyDegree{1_p},
                                                    degree0,
                                                    1_p,
                                                    degree2,
                                                    1_p);

          CHECK(result == correct);
        }

        SUBCASE("layer_norm_get_gamma_weights_parallel_shape") {
          SUBCASE("elementwise_affine = true") {
            ParallelTensorShape result =
                layer_norm_get_gamma_weights_parallel_shape(attrs_affine_true,
                                                            par_input);
            ParallelTensorShape correct = make_gamma_weights(
                SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, degree2);

            CHECK(result == correct);
          }

          SUBCASE("elementwise_affine = false") {
            CHECK_THROWS(layer_norm_get_gamma_weights_parallel_shape(
                attrs_affine_false, par_input));
          }
        }

        SUBCASE("layer_norm_get_beta_weights_parallel_shape") {
          SUBCASE("elementwise_affine = true") {
            ParallelTensorShape result =
                layer_norm_get_beta_weights_parallel_shape(attrs_affine_true,
                                                           par_input);
            ParallelTensorShape correct = make_beta_weights(
                SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, degree2);

            CHECK(result == correct);
          }

          SUBCASE("elementwise_affine = false") {
            CHECK_THROWS(layer_norm_get_beta_weights_parallel_shape(
                attrs_affine_false, par_input));
          }
        }
      }

      SUBCASE("partition parallelism (in axes)") {
        positive_int degree1 = 2_p;
        positive_int degree2 = 4_p;

        ParallelTensorShape par_input = make_input(
            SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, degree1, degree2, 1_p);

        SUBCASE("layer_norm_get_output_parallel_shape") {
          CHECK_THROWS(layer_norm_get_output_parallel_shape(attrs_affine_true,
                                                            par_input));
        }

        SUBCASE("layer_norm_get_gamma_weights_parallel_shape") {
          CHECK_THROWS(layer_norm_get_gamma_weights_parallel_shape(
              attrs_affine_true, par_input));
        }

        SUBCASE("layer_norm_get_beta_weights_parallel_shape") {
          CHECK_THROWS(layer_norm_get_beta_weights_parallel_shape(
              attrs_affine_true, par_input));
        }
      }

      SUBCASE("sum parallelism") {
        SumDegree sum_degree = SumDegree{2_p};

        ParallelTensorShape par_input =
            make_input(sum_degree, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p, 1_p);

        SUBCASE("layer_norm_get_output_parallel_shape") {
          CHECK_THROWS(layer_norm_get_output_parallel_shape(attrs_affine_true,
                                                            par_input));
        }

        SUBCASE("layer_norm_get_gamma_weights_parallel_shape") {
          CHECK_THROWS(layer_norm_get_gamma_weights_parallel_shape(
              attrs_affine_true, par_input));
        }

        SUBCASE("layer_norm_get_beta_weights_parallel_shape") {
          CHECK_THROWS(layer_norm_get_beta_weights_parallel_shape(
              attrs_affine_true, par_input));
        }
      }

      SUBCASE("discard copy parallelism") {
        DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_p};

        ParallelTensorShape par_input =
            make_input(SumDegree{1_p}, discard_copy_degree, 1_p, 1_p, 1_p, 1_p);

        SUBCASE("layer_norm_get_output_parallel_shape") {
          CHECK_THROWS(layer_norm_get_output_parallel_shape(attrs_affine_true,
                                                            par_input));
        }

        SUBCASE("layer_norm_get_gamma_weights_parallel_shape") {
          CHECK_THROWS(layer_norm_get_gamma_weights_parallel_shape(
              attrs_affine_true, par_input));
        }

        SUBCASE("layer_norm_get_beta_weights_parallel_shape") {
          CHECK_THROWS(layer_norm_get_beta_weights_parallel_shape(
              attrs_affine_true, par_input));
        }
      }
    }
  }
}
