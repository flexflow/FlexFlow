#include "op-attrs/ops/linear.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest.h"
#include "utils/integer_conversions.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_linear_incoming_tensor_roles(LinearAttrs)") {
    auto make_attrs = [](bool use_bias) {
      return LinearAttrs{
          /*out_channels=*/16,
          /*use_bias=*/use_bias,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };
    };

    SUBCASE("use_bias = true") {
      LinearAttrs attrs = make_attrs(/*use_bias=*/true);

      std::vector<IncomingTensorRole> result =
          get_linear_incoming_tensor_roles(attrs);
      std::vector<IncomingTensorRole> correct = {
          IncomingTensorRole::INPUT,
          IncomingTensorRole::WEIGHT,
          IncomingTensorRole::WEIGHT,
      };

      CHECK(result == correct);
    }

    SUBCASE("use_bias = false") {
      LinearAttrs attrs = make_attrs(/*use_bias=*/false);

      std::vector<IncomingTensorRole> result =
          get_linear_incoming_tensor_roles(attrs);
      std::vector<IncomingTensorRole> correct = {
          IncomingTensorRole::INPUT,
          IncomingTensorRole::WEIGHT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("Linear shape inference") {
    int out_channels = 16;
    LinearAttrs attrs = LinearAttrs{
        /*out_channels=*/out_channels,
        /*use_bias=*/true,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/Activation::RELU,
        /*regularizer=*/std::nullopt,
    };

    size_t batch_size = 12;
    size_t extra_dim = 16;
    size_t in_channels = 8;

    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                batch_size,
                extra_dim,
                in_channels,
            },
        },
        DataType::FLOAT,
    };

    TensorShape output = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                batch_size,
                extra_dim,
                size_t_from_int(out_channels),
            },
        },
        DataType::FLOAT,
    };

    TensorShape projection = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                in_channels,
                size_t_from_int(out_channels),
            },
        },
        DataType::FLOAT,
    };

    TensorShape bias = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
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

    // get_weight_shape
    {
      tl::expected<TensorShape, std::string> projection_result =
          get_projection_shape(attrs, input);
      tl::expected<TensorShape, std::string> projection_correct = projection;
      CHECK(projection_result == projection_correct);
    }

    // get_bias_shape
    {
      tl::expected<TensorShape, std::string> bias_result =
          get_bias_shape(attrs, input);
      tl::expected<TensorShape, std::string> bias_correct = bias;
      CHECK(bias_result == bias_correct);
    }

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          int o_batch,
                          int o_extra_dim,
                          int o_channel) {
      return lift_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered<int>{o_batch, o_extra_dim, o_channel});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           int o_batch,
                           int o_extra_dim,
                           int o_channel) {
      return lift_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered<int>{o_batch, o_extra_dim, o_channel});
    };

    auto make_projection = [&](SumDegree o_sum,
                               DiscardCopyDegree o_eq,
                               int o_inchannel,
                               int o_outchannel) {
      return lift_to_parallel_with_degrees(
          projection, o_sum, o_eq, FFOrdered<int>{o_inchannel, o_outchannel});
    };

    auto make_bias =
        [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o_outchannel) {
          return lift_to_parallel_with_degrees(
              bias, o_sum, o_eq, FFOrdered<int>{o_outchannel});
        };

    SUBCASE("data parallelism") {
      int input_sum_degree = 2;
      int extra_dim_degree = 8;
      int degree = 4;

      ParallelTensorShape par_input = make_input(SumDegree{input_sum_degree},
                                                 DiscardCopyDegree{1},
                                                 degree,
                                                 extra_dim_degree,
                                                 1);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_output(SumDegree{input_sum_degree},
                        DiscardCopyDegree{1},
                        degree,
                        extra_dim_degree,
                        1);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_projection_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_projection(
                SumDegree{1},
                DiscardCopyDegree{input_sum_degree * degree * extra_dim_degree},
                1,
                1);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_bias_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_bias(SumDegree{input_sum_degree},
                      DiscardCopyDegree{degree * extra_dim_degree},
                      1);
        CHECK(result == correct);
      }
    }

    SUBCASE("reduction parallelism") {
      int input_sum_degree = 2;
      int degree = 4;

      ParallelTensorShape par_input = make_input(
          SumDegree{input_sum_degree}, DiscardCopyDegree{1}, 1, 1, degree);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_output(SumDegree{input_sum_degree * degree},
                        DiscardCopyDegree{1},
                        1,
                        1,
                        1);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_projection_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_projection(
                SumDegree{1}, DiscardCopyDegree{input_sum_degree}, degree, 1);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_bias_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct = make_bias(
            SumDegree{input_sum_degree * degree}, DiscardCopyDegree{1}, 1);
        CHECK(result == correct);
      }
    }

    SUBCASE("output channel parallelism") {
      int input_sum_degree = 2;
      int degree = 4;

      ParallelTensorShape par_input = make_input(
          SumDegree{input_sum_degree}, DiscardCopyDegree{degree}, 1, 1, 1);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct = make_output(
            SumDegree{input_sum_degree}, DiscardCopyDegree{1}, 1, 1, degree);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_projection_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_projection(
                SumDegree{1}, DiscardCopyDegree{input_sum_degree}, 1, degree);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_bias_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct = make_bias(
            SumDegree{input_sum_degree}, DiscardCopyDegree{1}, degree);
        CHECK(result == correct);
      }
    }
  }
}
