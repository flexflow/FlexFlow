#include "kernels/batch_matmul_kernels_cpu.h"
#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("batch_matmul_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input_lhs =
        create_3d_accessor_r_with_contents<float>(
            {
                {
                    {3, 3, 6},
                    {2, 1, 5},
                    {1, 2, -2},
                    {8, 0.5, -3},
                },
                {
                    {5, 1, -2},
                    {-8, 0, -1},
                    {0.25, -0.3, -2},
                    {0, -1, -5},
                },
            },
            cpu_allocator);

    GenericTensorAccessorR input_rhs =
        create_3d_accessor_r_with_contents<float>(
            {
                {
                    {1.0f, 0.5f},
                    {2.0f, 4.0f},
                    {1.5f, -1.0f},
                },
                {{-6.0f, -0.5f}, {1.0f, -2.0f}, {0.4f, -3.5f}},
            },
            cpu_allocator);

    GenericTensorAccessorW result = create_zero_filled_accessor_w(
        TensorShape{
            TensorDims{FFOrdered{2_p, 4_p, 2_p}},
            DataType::FLOAT,
        },
        cpu_allocator);

    batch_matmul_cpu_forward_kernel(input_lhs, input_rhs, result);

    GenericTensorAccessorR correct = create_3d_accessor_r_with_contents<float>(
        {
            {
                {18.0, 7.5},
                {11.5, 0.0},
                {2.0, 10.5},
                {4.5, 9.0},
            },
            {
                {-29.8, 2.5},
                {47.6, 7.5},
                {-2.6, 7.475},
                {-3.0, 19.5},
            },
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result=", format_accessor_w_contents(result)));
  }

  TEST_CASE("batch_matmul_cpu_backward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input_lhs =
        create_3d_accessor_r_with_contents<float>(
            {
                {
                    {3, 3, 6},
                    {2, 1, 5},
                    {1, 2, -2},
                    {8, 0.5, -3},
                },
                {
                    {5, 1, -2},
                    {-8, 0, -1},
                    {0.25, -0.3, -2},
                    {0, -1, -5},
                },
            },
            cpu_allocator);

    GenericTensorAccessorR input_rhs =
        create_3d_accessor_r_with_contents<float>(
            {
                {
                    {1.0f, 0.5f},
                    {2.0f, 4.0f},
                    {1.5f, -1.0f},
                },
                {{-6.0f, -0.5f}, {1.0f, -2.0f}, {0.4f, -3.5f}},
            },
            cpu_allocator);

    GenericTensorAccessorR output = create_3d_accessor_r_with_contents<float>(
        {
            {
                {18.0, 7.5},
                {11.5, 0.0},
                {2.0, 10.5},
                {4.5, 9.0},
            },
            {
                {-29.8, 2.5},
                {47.6, 7.5},
                {-2.6, 7.475},
                {-3.0, 19.5},
            },
        },
        cpu_allocator);

    GenericTensorAccessorR output_grad =
        create_3d_accessor_r_with_contents<float>(
            {
                {
                    {1.0, 2.0},
                    {0.1, -1.0},
                    {0.5, 1.5},
                    {4.5, -3.0},
                },
                {
                    {0.0, -2.5},
                    {1.0, -0.5},
                    {-2.0, -0.5},
                    {-3.0, 0.25},
                },
            },
            cpu_allocator);

    GenericTensorAccessorW input_lhs_grad = create_zero_filled_accessor_w(
        TensorShape{
            TensorDims{FFOrdered{2_p, 4_p, 3_p}},
            DataType::FLOAT,
        },
        cpu_allocator);

    GenericTensorAccessorW input_rhs_grad = create_zero_filled_accessor_w(
        TensorShape{
            TensorDims{FFOrdered{2_p, 3_p, 2_p}},
            DataType::FLOAT,
        },
        cpu_allocator);

    batch_matmul_cpu_backward_kernel(
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input_lhs=*/input_lhs,
        /*input_lhs_grad=*/input_lhs_grad,
        /*input_rhs=*/input_rhs,
        /*input_rhs_grad=*/input_rhs_grad);

    SUBCASE("input_lhs_grad is correct") {
      GenericTensorAccessorW result = input_lhs_grad;

      GenericTensorAccessorR correct =
          create_3d_accessor_r_with_contents<float>(
              {
                  {
                      {2.0, 10.0, -0.5},
                      {-0.4, -3.8, 1.15},
                      {1.25, 7.0, -0.75},
                      {3.0, -3.0, 9.75},
                  },
                  {
                      {1.25, 5.0, 8.75},
                      {-5.75, 2.0, 2.15},
                      {12.25, -1.0, 0.95},
                      {17.875, -3.5, -2.075},
                  },
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result=", format_accessor_w_contents(result)));
    }

    SUBCASE("input_rhs_grad is correct") {
      GenericTensorAccessorW result = input_rhs_grad;

      GenericTensorAccessorR correct =
          create_3d_accessor_r_with_contents<float>(
              {
                  {
                      {39.7, -18.5},
                      {6.35, 6.5},
                      {-8.0, 13.0},
                  },
                  {
                      {-8.5, -8.625},
                      {3.6, -2.6},
                      {18.0, 5.25},
                  },
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result=", format_accessor_w_contents(result)));
    }
  }
}
