#include "kernels/element_binary_kernels_cpu.h"
#include "kernels/accessor.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("element_binary_cpu_forward_kernel") {
    SUBCASE("add") {
      Allocator cpu_allocator = create_local_cpu_memory_allocator();

      ElementBinaryAttrs attrs{
          /*op=*/ElementBinaryOp::ADD,
          /*compute_type=*/DataType::FLOAT,
          /*should_broadcast_lhs=*/false,
          /*should_broadcast_rhs=*/false,
      };

      GenericTensorAccessorR lhs = create_2d_accessor_r_with_contents<float>(
          {
              {1, 2, 3},
              {4, 5, 6},
              {7, 8, 9},
              {-1, -2, -3},
          },
          cpu_allocator);

      GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
          {
              {10, 50, -10},
              {20, 60, -20},
              {30, 70, -30},
              {40, 80, -40},
          },
          cpu_allocator);

      // Intentionally randomize this tensor so we can be confident we never read it
      GenericTensorAccessorW result =
          create_random_filled_accessor_w(lhs.shape, cpu_allocator);

      element_binary_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*lhs=*/lhs,
          /*rhs=*/rhs,
          /*output=*/result);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {11, 52, -7},
                  {24, 65, -14},
                  {37, 78, -21},
                  {39, 78, -43},
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }
  }

  TEST_CASE("element_binary_cpu_backward_kernel") {
    SUBCASE("add") {
      Allocator cpu_allocator = create_local_cpu_memory_allocator();

      ElementBinaryAttrs attrs{
          /*op=*/ElementBinaryOp::ADD,
          /*compute_type=*/DataType::FLOAT,
          /*should_broadcast_lhs=*/false,
          /*should_broadcast_rhs=*/false,
      };

      GenericTensorAccessorR lhs = create_2d_accessor_r_with_contents<float>(
          {
              {1, 2, 3},
              {4, 5, 6},
              {7, 8, 9},
              {-1, -2, -3},
          },
          cpu_allocator);

      GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
          {
              {10, 50, -10},
              {20, 60, -20},
              {30, 70, -30},
              {40, 80, -40},
          },
          cpu_allocator);

      GenericTensorAccessorW lhs_grad = create_zero_filled_accessor_w(
          get_tensor_shape_for_accessor_r(lhs), cpu_allocator);

      GenericTensorAccessorW rhs_grad = create_zero_filled_accessor_w(
          get_tensor_shape_for_accessor_r(rhs), cpu_allocator);

      GenericTensorAccessorR output = create_2d_accessor_r_with_contents<float>(
          {
              {11, 52, -7},
              {24, 65, -14},
              {73, 78, -21},
              {39, 78, -43},
          },
          cpu_allocator);

      GenericTensorAccessorR output_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, -1},
                  {6, 4, 2},
                  {0.5, 0.1, -2},
                  {0, 0.5, 0},
              },
              cpu_allocator);

      element_binary_cpu_backward_kernel(
          /*attrs=*/attrs,
          /*output=*/output,
          /*output_grad=*/output_grad,
          /*lhs=*/lhs,
          /*lhs_grad=*/lhs_grad,
          /*rhs=*/rhs,
          /*rhs_grad=*/rhs_grad);

      GenericTensorAccessorR correct_lhs_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, -1},
                  {6, 4, 2},
                  {0.5, 0.1, -2},
                  {0, 0.5, 0},
              },
              cpu_allocator);

      GenericTensorAccessorR correct_rhs_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, -1},
                  {6, 4, 2},
                  {0.5, 0.1, -2},
                  {0, 0.5, 0},
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(lhs_grad, correct_lhs_grad),
                    check_kv("lhs_grad", format_accessor_w_contents(lhs_grad)));

      CHECK_MESSAGE(accessors_are_equal(rhs_grad, correct_rhs_grad),
                    check_kv("rhs_grad", format_accessor_w_contents(rhs_grad)));
    }
  }
}
