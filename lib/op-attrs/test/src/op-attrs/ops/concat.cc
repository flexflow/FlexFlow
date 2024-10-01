#include "op-attrs/ops/concat.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(ConcatAttrs, std::vector<TensorShape>)") {
    ConcatAttrs attrs = ConcatAttrs{
        ff_dim_t{1},
    };

    SUBCASE("empty input shapes list passed") {
      std::vector<TensorShape> input_shapes = {};

      std::optional<TensorShape> result =
          optional_from_expected(get_output_shape(attrs, input_shapes));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    size_t dim0_size = 12;
    size_t dim2_size = 20;
    TensorShape input_shape1 = TensorShape{
        TensorDims{FFOrdered<size_t>{
            dim0_size,
            14,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    SUBCASE("single element input shapes list passed") {
      std::vector<TensorShape> input_shapes = {input_shape1};

      std::optional<TensorShape> result =
          optional_from_expected(get_output_shape(attrs, input_shapes));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    TensorShape input_shape2 = TensorShape{
        TensorDims{FFOrdered<size_t>{
            dim0_size,
            16,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    TensorShape input_shape3 = TensorShape{
        TensorDims{FFOrdered<size_t>{dim0_size, 18, dim2_size}},
        DataType::FLOAT,
    };

    SUBCASE("input shapes do not shared the same num_dims") {
      TensorShape mismatched_num_dims = TensorShape{
          TensorDims{FFOrdered<size_t>{
              dim0_size,
              20,
              dim2_size,
              1,
          }},
          DataType::FLOAT,
      };

      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3, mismatched_num_dims};

      std::optional<TensorShape> result =
          optional_from_expected(get_output_shape(attrs, input_shapes));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("concat axis is out of bounds") {
      attrs = ConcatAttrs{
          ff_dim_t{3},
      };

      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3};

      std::optional<TensorShape> result =
          optional_from_expected(get_output_shape(attrs, input_shapes));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input shapes are valid") {
      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3};

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_shapes);
      tl::expected<TensorShape, std::string> correct = TensorShape{
          TensorDims{FFOrdered<size_t>{
              dim0_size,
              14 + 16 + 18,
              dim2_size,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_output_shape(ConcatAttrs, std::vector<ParallelTensorShape>)") {
    ConcatAttrs attrs = ConcatAttrs{
        ff_dim_t{1},
    };

    size_t dim0_size = 12;
    size_t dim2_size = 20;

    TensorShape input_shape1 = TensorShape{
        TensorDims{FFOrdered<size_t>{
            dim0_size,
            14,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    TensorShape input_shape2 = TensorShape{
        TensorDims{FFOrdered<size_t>{
            dim0_size,
            16,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    TensorShape input_shape3 = TensorShape{
        TensorDims{FFOrdered<size_t>{dim0_size, 18, dim2_size}},
        DataType::FLOAT,
    };

    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered<size_t>{dim0_size, 14 + 16 + 18, dim2_size}},
        DataType::FLOAT,
    };

    auto lift_input1 =
        [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o0, int o1, int o2) {
          return lift_to_parallel_with_degrees(
              input_shape1, o_sum, o_eq, FFOrdered<int>{o0, o1, o2});
        };

    auto lift_input2 =
        [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o0, int o1, int o2) {
          return lift_to_parallel_with_degrees(
              input_shape2, o_sum, o_eq, FFOrdered<int>{o0, o1, o2});
        };

    auto lift_input3 =
        [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o0, int o1, int o2) {
          return lift_to_parallel_with_degrees(
              input_shape3, o_sum, o_eq, FFOrdered<int>{o0, o1, o2});
        };

    auto lift_output =
        [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o0, int o1, int o2) {
          return lift_to_parallel_with_degrees(
              output_shape, o_sum, o_eq, FFOrdered<int>{o0, o1, o2});
        };

    SUBCASE("sum reduction parallelism") {
      SUBCASE("matching") {
        SumDegree sum_degree = SumDegree{2};

        std::vector<ParallelTensorShape> inputs = {
            lift_input1(sum_degree, DiscardCopyDegree{1}, 1, 1, 1),
            lift_input2(sum_degree, DiscardCopyDegree{1}, 1, 1, 1),
            lift_input3(sum_degree, DiscardCopyDegree{1}, 1, 1, 1),
        };

        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, inputs);
        tl::expected<ParallelTensorShape, std::string> correct =
            lift_output(sum_degree, DiscardCopyDegree{1}, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorShape> inputs = {
            lift_input1(SumDegree{2}, DiscardCopyDegree{1}, 1, 1, 1),
            lift_input2(SumDegree{4}, DiscardCopyDegree{1}, 1, 1, 1),
            lift_input3(SumDegree{4}, DiscardCopyDegree{1}, 1, 1, 1),
        };

        std::optional<ParallelTensorShape> result =
            optional_from_expected(get_output_shape(attrs, inputs));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("discard copy reduction parallelism") {
      SUBCASE("matching") {
        DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2};

        std::vector<ParallelTensorShape> inputs = {
            lift_input1(SumDegree{1}, discard_copy_degree, 1, 1, 1),
            lift_input2(SumDegree{1}, discard_copy_degree, 1, 1, 1),
            lift_input3(SumDegree{1}, discard_copy_degree, 1, 1, 1),
        };

        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, inputs);
        tl::expected<ParallelTensorShape, std::string> correct =
            lift_output(SumDegree{1}, discard_copy_degree, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorShape> inputs = {
            lift_input1(SumDegree{1}, DiscardCopyDegree{2}, 1, 1, 1),
            lift_input2(SumDegree{1}, DiscardCopyDegree{2}, 1, 1, 1),
            lift_input3(SumDegree{1}, DiscardCopyDegree{4}, 1, 1, 1),
        };

        std::optional<ParallelTensorShape> result =
            optional_from_expected(get_output_shape(attrs, inputs));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("parallelism in axis dim") {
      SUBCASE("matching") {
        int degree = 2;

        std::vector<ParallelTensorShape> inputs = {
            lift_input1(SumDegree{1}, DiscardCopyDegree{1}, 1, degree, 1),
            lift_input2(SumDegree{1}, DiscardCopyDegree{1}, 1, degree, 1),
            lift_input3(SumDegree{1}, DiscardCopyDegree{1}, 1, degree, 1),
        };

        std::optional<ParallelTensorShape> result =
            optional_from_expected(get_output_shape(attrs, inputs));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorShape> inputs = {
            lift_input1(SumDegree{1}, DiscardCopyDegree{1}, 1, 1, 1),
            lift_input2(SumDegree{1}, DiscardCopyDegree{1}, 1, 1, 1),
            lift_input3(SumDegree{1}, DiscardCopyDegree{1}, 1, 2, 1),
        };

        std::optional<ParallelTensorShape> result =
            optional_from_expected(get_output_shape(attrs, inputs));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("parallelism in non-axis shard dims") {
      SUBCASE("matching") {
        int degree0 = 2;
        int degree2 = 4;

        std::vector<ParallelTensorShape> inputs = {
            lift_input1(
                SumDegree{1}, DiscardCopyDegree{1}, degree0, 1, degree2),
            lift_input2(
                SumDegree{1}, DiscardCopyDegree{1}, degree0, 1, degree2),
            lift_input3(
                SumDegree{1}, DiscardCopyDegree{1}, degree0, 1, degree2),
        };

        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, inputs);
        tl::expected<ParallelTensorShape, std::string> correct = lift_output(
            SumDegree{1}, DiscardCopyDegree{1}, degree0, 1, degree2);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorShape> inputs = {
            lift_input1(SumDegree{1}, DiscardCopyDegree{1}, 2, 1, 4),
            lift_input2(SumDegree{1}, DiscardCopyDegree{1}, 4, 1, 2),
            lift_input3(SumDegree{1}, DiscardCopyDegree{1}, 4, 1, 2),
        };

        std::optional<ParallelTensorShape> result =
            optional_from_expected(get_output_shape(attrs, inputs));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("parallelism degrees are not mutually exclusive") {
      SumDegree sum_degree = SumDegree{3};
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{5};
      int degree0 = 2;
      int degree2 = 4;

      std::vector<ParallelTensorShape> inputs = {
          lift_input1(sum_degree, discard_copy_degree, degree0, 1, degree2),
          lift_input2(sum_degree, discard_copy_degree, degree0, 1, degree2),
          lift_input3(sum_degree, discard_copy_degree, degree0, 1, degree2),
      };

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, inputs);
      tl::expected<ParallelTensorShape, std::string> correct =
          lift_output(sum_degree, discard_copy_degree, degree0, 1, degree2);

      CHECK(result == correct);
    }
  }
}
