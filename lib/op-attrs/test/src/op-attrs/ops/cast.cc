#include "op-attrs/ops/cast.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cast_get_output_shape") {
    CastAttrs attrs = CastAttrs{DataType::DOUBLE};

    TensorShape input = TensorShape{
        TensorDims{FFOrdered{12_p, 16_p}},
        DataType::FLOAT,
    };

    TensorShape result = cast_get_output_shape(attrs, input);

    TensorShape correct = TensorShape{
        TensorDims{FFOrdered{12_p, 16_p}},
        DataType::DOUBLE,
    };

    CHECK(result == correct);
  }

  TEST_CASE("cast_get_output_parallel_shape") {
    CastAttrs attrs = CastAttrs{DataType::DOUBLE};

    TensorShape input = TensorShape{
        TensorDims{FFOrdered{12_p, 16_p}},
        DataType::FLOAT,
    };

    TensorShape output = TensorShape{
        TensorDims{FFOrdered{12_p, 16_p}},
        DataType::DOUBLE,
    };

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          positive_int o_batch,
                          positive_int o_features) -> ParallelTensorShape {
      return lift_shape_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered{o_batch, o_features});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o_batch,
                           positive_int o_outchannels) -> ParallelTensorShape {
      return lift_shape_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered{o_batch, o_outchannels});
    };

    SumDegree sum_degree = SumDegree{2_p};
    DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{3_p};
    positive_int batch_degree = 4_p;
    positive_int feature_degree = 8_p;
    ParallelTensorShape par_input = make_input(
        sum_degree, discard_copy_degree, batch_degree, feature_degree);

    ParallelTensorShape result =
        cast_get_output_parallel_shape(attrs, par_input);

    ParallelTensorShape correct = make_output(
        sum_degree, discard_copy_degree, batch_degree, feature_degree);

    CHECK(result == correct);
  }
}
