#include "op-attrs/ops/softmax.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(SoftmaxAttrs, TensorShape)") {
    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            12_p,
            14_p,
            16_p,
        }},
        DataType::FLOAT,
    };

    SUBCASE("attrs.dim in bounds") {
      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      TensorShape result = get_output_shape(attrs, input);
      TensorShape correct = input;

      CHECK(result == correct);
    }

    SUBCASE("attrs.dims out of bounds") {
      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{4_n}};

      CHECK_THROWS(get_output_shape(attrs, input));
    }
  }

  TEST_CASE("get_output_shape(SoftmaxAttrs, ParallelTensorShape)") {
    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            12_p,
            14_p,
            16_p,
        }},
        DataType::FLOAT,
    };
    TensorShape output = input;

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          positive_int o0,
                          positive_int o1,
                          positive_int o2) {
      return lift_shape_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered{o0, o1, o2});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o0,
                           positive_int o1,
                           positive_int o2) {
      return lift_shape_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered{o0, o1, o2});
    };

    SUBCASE("partition parallelism in non-softmax-dim (valid)") {
      positive_int degree0 = 2_p;
      positive_int degree2 = 4_p;

      ParallelTensorShape par_input = make_input(
          SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2);

      SUBCASE("attrs.dim in bounds") {
        SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

        ParallelTensorShape result = get_output_shape(attrs, par_input);
        ParallelTensorShape correct = make_output(
            SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2);

        CHECK(result == correct);
      }

      SUBCASE("attrs.dims out of bounds") {
        SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{4_n}};

        CHECK_THROWS(get_output_shape(attrs, par_input));
      }
    }

    SUBCASE("partition parallism in softmax dim (invalid)") {
      positive_int degree1 = 2_p;

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      ParallelTensorShape par_input =
          make_input(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, degree1, 1_p);

      CHECK_THROWS(get_output_shape(attrs, par_input));
    }

    SUBCASE("sum parallelism (invalid)") {
      SumDegree sum_degree = SumDegree{2_p};

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      ParallelTensorShape par_input =
          make_input(sum_degree, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);

      CHECK_THROWS(get_output_shape(attrs, par_input));
    }

    SUBCASE("discard copy parallelism (invalid)") {
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_p};

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      ParallelTensorShape par_input =
          make_input(SumDegree{1_p}, discard_copy_degree, 1_p, 1_p, 1_p);

      CHECK_THROWS(get_output_shape(attrs, par_input));
    }
  }
}
