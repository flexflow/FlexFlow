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

    size_t d1 = 12;
    size_t d2 = 16;
    TensorShape input = TensorShape{
        TensorDims{FFOrdered<size_t>{d1, d2}},
        input_datatype,
    };

    TensorShape output = TensorShape{
        TensorDims{FFOrdered<size_t>{d1, d2}},
        output_datatype,
    };

    SUBCASE("get_output_shape(CastAttrs, TensorShape)") {
      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input);
      tl::expected<TensorShape, std::string> correct = output;
      CHECK(result == correct);
    }

    SUBCASE("get_output_shape(CastAttrs, ParallelTensorShape)") {
      auto make_input = [&](SumDegree o_sum,
                            DiscardCopyDegree o_eq,
                            int o_batch,
                            int o_features) {
        return lift_to_parallel_with_degrees(
            input, o_sum, o_eq, FFOrdered<int>{o_batch, o_features});
      };

      auto make_output = [&](SumDegree o_sum,
                             DiscardCopyDegree o_eq,
                             int o_batch,
                             int o_outchannels) {
        return lift_to_parallel_with_degrees(
            output, o_sum, o_eq, FFOrdered<int>{o_batch, o_outchannels});
      };

      SumDegree sum_degree = SumDegree{2};
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{3};
      int batch_degree = 4;
      int feature_degree = 8;
      ParallelTensorShape par_input = make_input(
          sum_degree, discard_copy_degree, batch_degree, feature_degree);

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, par_input);
      tl::expected<ParallelTensorShape, std::string> correct = make_output(
          sum_degree, discard_copy_degree, batch_degree, feature_degree);

      CHECK(result == correct);
    }
  }
}
