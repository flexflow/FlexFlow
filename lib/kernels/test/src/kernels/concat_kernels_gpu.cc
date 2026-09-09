#include "kernels/concat_kernels_gpu.h"
#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("concat_gpu_forward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    std::vector<GenericTensorAccessorR> inputs = {
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 2, 3},
                {4, 5, 6},
            },
            allocator),
        create_2d_accessor_r_with_contents<float>(
            {
                {-1, -2, -3},
                {-4, -5, -6},
            },
            allocator),
        create_2d_accessor_r_with_contents<float>(
            {
                {0.5, 0.25, 0.125},
                {10, 20, 30},
            },
            allocator),
    };

    SUBCASE("axis = 0") {
      ConcatAttrs attrs = ConcatAttrs{
          /*axis=*/ff_dim_t{0_n},
          /*num_inputs=*/int_ge_two{3},
      };

      TensorShape output_shape = TensorShape{
          TensorDims{FFOrdered{6_p, 3_p}},
          DataType::FLOAT,
      };

      // Intentionally randomize this tensor so we can be confident we never
      // read it
      GenericTensorAccessorW output =
          create_random_filled_accessor_w(output_shape, allocator);

      concat_gpu_forward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*attrs=*/attrs,
          /*inputs=*/inputs,
          /*output=*/output);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, 3},
                  {4, 5, 6},
                  {-1, -2, -3},
                  {-4, -5, -6},
                  {0.5, 0.25, 0.125},
                  {10, 20, 30},
              },
              allocator);

      CHECK_MESSAGE(accessors_are_equal(output, correct),
                    check_kv("output", format_accessor_w_contents(output)));
    }

    SUBCASE("axis = 1") {
      ConcatAttrs attrs = ConcatAttrs{
          /*axis=*/ff_dim_t{1_n},
          /*num_inputs=*/int_ge_two{3},
      };

      TensorShape output_shape = TensorShape{
          TensorDims{FFOrdered{2_p, 9_p}},
          DataType::FLOAT,
      };

      // Intentionally randomize this tensor so we can be confident we never
      // read it
      GenericTensorAccessorW output =
          create_random_filled_accessor_w(output_shape, allocator);

      concat_gpu_forward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*attrs=*/attrs,
          /*inputs=*/inputs,
          /*output=*/output);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, 3, -1, -2, -3, 0.5, 0.25, 0.125},
                  {4, 5, 6, -4, -5, -6, 10, 20, 30},
              },
              allocator);

      CHECK_MESSAGE(accessors_are_equal(output, correct),
                    check_kv("output", format_accessor_w_contents(output)));
    }
  }

  TEST_CASE("concat_gpu_backward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    std::vector<GenericTensorAccessorR> inputs = {
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 2, 3},
                {4, 5, 6},
            },
            allocator),
        create_2d_accessor_r_with_contents<float>(
            {
                {-1, -2, -3},
                {-4, -5, -6},
            },
            allocator),
        create_2d_accessor_r_with_contents<float>(
            {
                {0.5, 0.25, 0.125},
                {10, 20, 30},
            },
            allocator),
    };

    // input_grads are accumulated into, so they need to start from known values
    auto make_input_grads = [&]() {
      return std::vector<GenericTensorAccessorW>{
          create_2d_accessor_w_with_contents<float>(
              {
                  {100, 101, 102},
                  {103, 104, 105},
              },
              allocator),
          create_2d_accessor_w_with_contents<float>(
              {
                  {200, 201, 202},
                  {203, 204, 205},
              },
              allocator),
          create_2d_accessor_w_with_contents<float>(
              {
                  {300, 301, 302},
                  {303, 304, 305},
              },
              allocator),
      };
    };

    SUBCASE("axis = 0") {
      ConcatAttrs attrs = ConcatAttrs{
          /*axis=*/ff_dim_t{0_n},
          /*num_inputs=*/int_ge_two{3},
      };

      GenericTensorAccessorR output = create_2d_accessor_r_with_contents<float>(
          {
              {1, 2, 3},
              {4, 5, 6},
              {-1, -2, -3},
              {-4, -5, -6},
              {0.5, 0.25, 0.125},
              {10, 20, 30},
          },
          allocator);

      GenericTensorAccessorR output_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {-1.75, -1.5, -1.25},
                  {-1, -0.75, -0.5},
                  {-0.25, 0, 0.25},
                  {0.5, 0.75, 1},
                  {1.25, 1.5, 1.75},
                  {2, 2.25, 2.5},
              },
              allocator);

      std::vector<GenericTensorAccessorW> input_grads = make_input_grads();

      concat_gpu_backward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*attrs=*/attrs,
          /*output=*/output,
          /*output_grad=*/output_grad,
          /*inputs=*/inputs,
          /*input_grads=*/input_grads);

      std::vector<GenericTensorAccessorR> correct = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {98.25, 99.5, 100.75},
                  {102, 103.25, 104.5},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {199.75, 201, 202.25},
                  {203.5, 204.75, 206},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {301.25, 302.5, 303.75},
                  {305, 306.25, 307.5},
              },
              allocator),
      };

      for (int i = 0; i < input_grads.size(); i++) {
        CHECK_MESSAGE(accessors_are_equal(input_grads.at(i), correct.at(i)),
                      check_kv("input_grad",
                               format_accessor_w_contents(input_grads.at(i))));
      }
    }

    SUBCASE("axis = 1") {
      ConcatAttrs attrs = ConcatAttrs{
          /*axis=*/ff_dim_t{1_n},
          /*num_inputs=*/int_ge_two{3},
      };

      GenericTensorAccessorR output = create_2d_accessor_r_with_contents<float>(
          {
              {1, 2, 3, -1, -2, -3, 0.5, 0.25, 0.125},
              {4, 5, 6, -4, -5, -6, 10, 20, 30},
          },
          allocator);

      GenericTensorAccessorR output_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {-1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25},
                  {0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5},
              },
              allocator);

      std::vector<GenericTensorAccessorW> input_grads = make_input_grads();

      concat_gpu_backward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*attrs=*/attrs,
          /*output=*/output,
          /*output_grad=*/output_grad,
          /*inputs=*/inputs,
          /*input_grads=*/input_grads);

      std::vector<GenericTensorAccessorR> correct = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {98.25, 99.5, 100.75},
                  {103.5, 104.75, 106},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {199, 200.25, 201.5},
                  {204.25, 205.5, 206.75},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {299.75, 301, 302.25},
                  {305, 306.25, 307.5},
              },
              allocator),
      };

      for (int i = 0; i < input_grads.size(); i++) {
        CHECK_MESSAGE(accessors_are_equal(input_grads.at(i), correct.at(i)),
                      check_kv("input_grad",
                               format_accessor_w_contents(input_grads.at(i))));
      }
    }
  }
}
