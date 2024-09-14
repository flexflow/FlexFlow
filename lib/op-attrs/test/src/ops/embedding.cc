#include "op-attrs/ops/embedding.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Sum embedding shape inference") {
    int out_channels = 128;
    int num_entries = 1024;
    EmbeddingAttrs attrs = EmbeddingAttrs{
        /*num_entries=*/num_entries,
        /*out_channels=*/out_channels,
        /*aggr=*/AggregateOp::SUM,
        /*data_type=*/DataType::FLOAT,
    };

    size_t batch_size = 48;
    size_t features_dim = 56;

    TensorShape input = TensorShape{
        TensorDims{FFOrdered<size_t>{
            batch_size,
            features_dim,
        }},
        DataType::INT32,
    };

    TensorShape output = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                batch_size,
                size_t_from_int(out_channels),
            },
        },
        DataType::FLOAT,
    };

    TensorShape weights = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                size_t_from_int(num_entries),
                size_t_from_int(out_channels),
            },
        },
        DataType::FLOAT,
    };

    // get_output_shape
    {
      tl::expected<TensorShape, std::string> output_result =
          get_output_shape(attrs, input);
      tl::expected<TensorShape, std::string> output_correct = output;
      CHECK(output_result == output_correct);
    }

    // get_weights_shape
    {
      tl::expected<TensorShape, std::string> weight_result =
          get_weights_shape(attrs, input);
      tl::expected<TensorShape, std::string> weight_correct = weights;
      CHECK(weight_result == weight_correct);
    }

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

    auto make_weights = [&](SumDegree o_sum,
                            DiscardCopyDegree o_eq,
                            int o_entries,
                            int o_outchannels) {
      return lift_to_parallel_with_degrees(
          weights, o_sum, o_eq, FFOrdered<int>{o_entries, o_outchannels});
    };

    SUBCASE("data parallelism") {
      int degree = 4;
      ParallelTensorShape par_input =
          make_input(SumDegree{1}, DiscardCopyDegree{1}, degree, 1);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_output(SumDegree{1}, DiscardCopyDegree{1}, degree, 1);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_weights_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_weights(SumDegree{1}, DiscardCopyDegree{degree}, 1, 1);
        CHECK(result == correct);
      }
    }

    SUBCASE("input features parallelism") {
      int degree = 4;
      ParallelTensorShape input =
          make_input(SumDegree{1}, DiscardCopyDegree{1}, 1, degree);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_output(SumDegree{degree}, DiscardCopyDegree{1}, 1, 1);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_weights_shape(attrs, input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_weights(SumDegree{1}, DiscardCopyDegree{degree}, 1, 1);
        CHECK(result == correct);
      }
    }

    SUBCASE("output channel shard parallelism") {
      // NOTE (@lockshaw): in the current (parallel shape inference from just
      // input tensor) representation we have to choose between either
      // parallelism in the weight channel dimension or in the weight entry
      // dimension. For now we choose to represent parallelism in the channel
      // dimension, but partitioning in the entry dimension is also potentially
      // useful as it produces sum parallelism in the output
      int degree = 4;
      ParallelTensorShape input =
          make_input(SumDegree{1}, DiscardCopyDegree{degree}, 1, 1);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_output(SumDegree{1}, DiscardCopyDegree{1}, 1, degree);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_weights_shape(attrs, input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_weights(SumDegree{1}, DiscardCopyDegree{1}, 1, degree);
        CHECK(result == correct);
      }
    }
  }
}
