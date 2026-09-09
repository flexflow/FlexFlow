#include "kernels/split_kernels_cpu.h"
#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static GenericTensorAccessorR make_input(Allocator &allocator) {
  return create_2d_accessor_r_with_contents<float>(
      {
          {1, 2, 3, 4, 5, 6},
          {-1, -2, -3, -4, -5, -6},
          {0.5, 0.25, 0.125, 10, 20, 30},
          {7, 8, 9, 0.75, 0.5, 0.25},
      },
      allocator);
}

// input_grad is accumulated into, so it needs to start from a known value
static GenericTensorAccessorW make_input_grad(Allocator &allocator) {
  return create_2d_accessor_w_with_contents<float>(
      {
          {-3, -1, 1, 3, 5, 7},
          {9, 11, 13, 15, 17, 19},
          {21, 23, 25, 27, 29, 31},
          {33, 35, 37, 39, 41, 43},
      },
      allocator);
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("split_cpu_forward_kernel") {
    Allocator allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = make_input(allocator);

    SUBCASE("axis = 0") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/{1_p, 3_p},
          /*axis=*/ff_dim_t{0_n},
      };

      // Intentionally randomize these tensors so we can be confident we never
      // read them
      std::vector<GenericTensorAccessorW> outputs = {
          create_random_filled_accessor_w(
              TensorShape{TensorDims{FFOrdered{1_p, 6_p}}, DataType::FLOAT},
              allocator),
          create_random_filled_accessor_w(
              TensorShape{TensorDims{FFOrdered{3_p, 6_p}}, DataType::FLOAT},
              allocator),
      };

      split_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input,
          /*outputs=*/outputs);

      std::vector<GenericTensorAccessorR> correct = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, 3, 4, 5, 6},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {-1, -2, -3, -4, -5, -6},
                  {0.5, 0.25, 0.125, 10, 20, 30},
                  {7, 8, 9, 0.75, 0.5, 0.25},
              },
              allocator),
      };

      for (int i = 0; i < outputs.size(); i++) {
        CHECK_MESSAGE(
            accessors_are_equal(outputs.at(i), correct.at(i)),
            check_kv("output", format_accessor_w_contents(outputs.at(i))));
      }
    }

    SUBCASE("axis = 1") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/{2_p, 1_p, 3_p},
          /*axis=*/ff_dim_t{1_n},
      };

      // Intentionally randomize these tensors so we can be confident we never
      // read them
      std::vector<GenericTensorAccessorW> outputs = {
          create_random_filled_accessor_w(
              TensorShape{TensorDims{FFOrdered{4_p, 2_p}}, DataType::FLOAT},
              allocator),
          create_random_filled_accessor_w(
              TensorShape{TensorDims{FFOrdered{4_p, 1_p}}, DataType::FLOAT},
              allocator),
          create_random_filled_accessor_w(
              TensorShape{TensorDims{FFOrdered{4_p, 3_p}}, DataType::FLOAT},
              allocator),
      };

      split_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input,
          /*outputs=*/outputs);

      std::vector<GenericTensorAccessorR> correct = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2},
                  {-1, -2},
                  {0.5, 0.25},
                  {7, 8},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {3},
                  {-3},
                  {0.125},
                  {9},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {4, 5, 6},
                  {-4, -5, -6},
                  {10, 20, 30},
                  {0.75, 0.5, 0.25},
              },
              allocator),
      };

      for (int i = 0; i < outputs.size(); i++) {
        CHECK_MESSAGE(
            accessors_are_equal(outputs.at(i), correct.at(i)),
            check_kv("output", format_accessor_w_contents(outputs.at(i))));
      }
    }
  }

  TEST_CASE("split_cpu_backward_kernel") {
    Allocator allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = make_input(allocator);

    SUBCASE("axis = 0") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/{1_p, 3_p},
          /*axis=*/ff_dim_t{0_n},
      };

      std::vector<GenericTensorAccessorR> outputs = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, 3, 4, 5, 6},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {-1, -2, -3, -4, -5, -6},
                  {0.5, 0.25, 0.125, 10, 20, 30},
                  {7, 8, 9, 0.75, 0.5, 0.25},
              },
              allocator),
      };

      std::vector<GenericTensorAccessorR> output_grads = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {-0.75, -0.5, -0.25, 0, 0.25, 0.5},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {9.25, 9.5, 9.75, 10, 10.25, 10.5},
                  {10.75, 11, 11.25, 11.5, 11.75, 12},
                  {12.25, 12.5, 12.75, 13, 13.25, 13.5},
              },
              allocator),
      };

      GenericTensorAccessorW input_grad = make_input_grad(allocator);

      split_cpu_backward_kernel(
          /*attrs=*/attrs,
          /*outputs=*/outputs,
          /*output_grads=*/output_grads,
          /*input=*/input,
          /*input_grad=*/input_grad);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {-3.75, -1.5, 0.75, 3, 5.25, 7.5},
                  {18.25, 20.5, 22.75, 25, 27.25, 29.5},
                  {31.75, 34, 36.25, 38.5, 40.75, 43},
                  {45.25, 47.5, 49.75, 52, 54.25, 56.5},
              },
              allocator);

      CHECK_MESSAGE(
          accessors_are_equal(input_grad, correct),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }

    SUBCASE("axis = 1") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/{2_p, 1_p, 3_p},
          /*axis=*/ff_dim_t{1_n},
      };

      std::vector<GenericTensorAccessorR> outputs = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2},
                  {-1, -2},
                  {0.5, 0.25},
                  {7, 8},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {3},
                  {-3},
                  {0.125},
                  {9},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {4, 5, 6},
                  {-4, -5, -6},
                  {10, 20, 30},
                  {0.75, 0.5, 0.25},
              },
              allocator),
      };

      std::vector<GenericTensorAccessorR> output_grads = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {-0.75, -0.5},
                  {-0.25, 0},
                  {0.25, 0.5},
                  {0.75, 1},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {9.25},
                  {9.5},
                  {9.75},
                  {10},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {19.25, 19.5, 19.75},
                  {20, 20.25, 20.5},
                  {20.75, 21, 21.25},
                  {21.5, 21.75, 22},
              },
              allocator),
      };

      GenericTensorAccessorW input_grad = make_input_grad(allocator);

      split_cpu_backward_kernel(
          /*attrs=*/attrs,
          /*outputs=*/outputs,
          /*output_grads=*/output_grads,
          /*input=*/input,
          /*input_grad=*/input_grad);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {-3.75, -1.5, 10.25, 22.25, 24.5, 26.75},
                  {8.75, 11, 22.5, 35, 37.25, 39.5},
                  {21.25, 23.5, 34.75, 47.75, 50, 52.25},
                  {33.75, 36, 47, 60.5, 62.75, 65},
              },
              allocator);

      CHECK_MESSAGE(
          accessors_are_equal(input_grad, correct),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }
  }
}
