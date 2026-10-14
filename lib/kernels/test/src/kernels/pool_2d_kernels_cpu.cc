#include "kernels/pool_2d_kernels_cpu.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include "utils/not_implemented.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("pool2d_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3, 6},
            {2, 1, 5},
            {1, 2, -2},
            {8, 0.5, -3},
        },
        cpu_allocator);

    SUBCASE("max pool") {
      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {3, 6},
                  {2, 5},
                  {8, 2},
              },
              cpu_allocator);

      GenericTensorAccessorW result =
          create_zero_filled_accessor_w(correct.shape, cpu_allocator);

      Pool2DAttrs attrs = Pool2DAttrs{
          /*kernel_h=*/2_p,
          /*kernel_w=*/2_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*pool_type=*/PoolOp::MAX,
          /*activation=*/std::nullopt,
      };

      pool2d_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/result);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("avg pool") {
      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {2.25, 3.75},
                  {1.5, 1.5},
                  {2.875, -0.625},
              },
              cpu_allocator);

      GenericTensorAccessorW result =
          create_zero_filled_accessor_w(correct.shape, cpu_allocator);

      Pool2DAttrs attrs = Pool2DAttrs{
          /*kernel_h=*/2_p,
          /*kernel_w=*/2_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*pool_type=*/PoolOp::AVG,
          /*activation=*/std::nullopt,
      };

      pool2d_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/result);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("asymmetric filter") {
      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {3, 6},
                  {8, 5},
              },
              cpu_allocator);

      GenericTensorAccessorW result =
          create_zero_filled_accessor_w(correct.shape, cpu_allocator);

      Pool2DAttrs attrs = Pool2DAttrs{
          /*kernel_h=*/3_p,
          /*kernel_w=*/2_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*pool_type=*/PoolOp::MAX,
          /*activation=*/std::nullopt,
      };

      pool2d_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/result);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("stride > 1") {
      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {3, 6},
                  {8, 2},
              },
              cpu_allocator);

      GenericTensorAccessorW result =
          create_zero_filled_accessor_w(correct.shape, cpu_allocator);

      Pool2DAttrs attrs = Pool2DAttrs{
          /*kernel_h=*/2_p,
          /*kernel_w=*/2_p,
          /*stride_h=*/2_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*pool_type=*/PoolOp::MAX,
          /*activation=*/std::nullopt,
      };

      pool2d_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/result);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("padding > 1") {
      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {3, 3, 6, 6},
                  {2, 2, 5, 5},
                  {8, 8, 2, 0},
              },
              cpu_allocator);

      GenericTensorAccessorW result =
          create_zero_filled_accessor_w(correct.shape, cpu_allocator);

      Pool2DAttrs attrs = Pool2DAttrs{
          /*kernel_h=*/2_p,
          /*kernel_w=*/2_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/1_n,
          /*pool_type=*/PoolOp::MAX,
          /*activation=*/std::nullopt,
      };

      pool2d_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/result);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }
  }
}
