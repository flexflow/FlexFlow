#include "op-attrs/ops/cast.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Cast shape inference") {
    DataType input_datatype = DataType::FLOAT;
    DataType output_datatype = DataType::DOUBLE;

    CastAttrs attrs = CastAttrs{output_datatype};

    positive_int d1 = 12_p;
    positive_int d2 = 16_p;
    TensorShape input = TensorShape{
        TensorDims{FFOrdered{d1, d2}},
        input_datatype,
    };

    TensorShape output = TensorShape{
        TensorDims{FFOrdered{d1, d2}},
        output_datatype,
    };

    SUBCASE("get_output_shape(CastAttrs, TensorShape)") {
      TensorShape result = get_output_shape(attrs, input);
      TensorShape correct = output;
      CHECK(result == correct);
    }

    SUBCASE("get_output_shape(CastAttrs, ParallelTensorShape)") {
      auto make_input = [&](SumDegree o_sum,
                            DiscardCopyDegree o_eq,
                            positive_int o_batch,
                            positive_int o_features) {
        return lift_shape_to_parallel_with_degrees(
            input, o_sum, o_eq, FFOrdered{o_batch, o_features});
      };

      auto make_output = [&](SumDegree o_sum,
                             DiscardCopyDegree o_eq,
                             positive_int o_batch,
                             positive_int o_outchannels) {
        return lift_shape_to_parallel_with_degrees(
            output, o_sum, o_eq, FFOrdered{o_batch, o_outchannels});
      };

      SumDegree sum_degree = SumDegree{2_p};
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{3_p};
      positive_int batch_degree = 4_p;
      positive_int feature_degree = 8_p;
      ParallelTensorShape par_input = make_input(
          sum_degree, discard_copy_degree, batch_degree, feature_degree);

      ParallelTensorShape result = get_output_shape(attrs, par_input);
      ParallelTensorShape correct = make_output(
          sum_degree, discard_copy_degree, batch_degree, feature_degree);

      CHECK(result == correct);
    }
  }
}
