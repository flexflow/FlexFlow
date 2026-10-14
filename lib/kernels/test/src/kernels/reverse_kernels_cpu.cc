#include "kernels/reverse_kernels_cpu.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Reverse::cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_3d_accessor_r_with_contents<int32_t>(
        {
            {
                {1, 3, 2},
                {4, 2, 1},
            },
            {
                {3, 3, 6},
                {2, 1, 5},
            },
        },
        cpu_allocator);

    GenericTensorAccessorW result = create_zero_filled_accessor_w(
        TensorShape{
            TensorDims{FFOrdered{2_p, 2_p, 3_p}},
            DataType::INT32,
        },
        cpu_allocator);

    SUBCASE("axis = ff_dim_t{0}") {
      ReverseAttrs attrs = ReverseAttrs{
          /*axis=*/ff_dim_t{0_n},
      };

      GenericTensorAccessorR correct =
          create_3d_accessor_r_with_contents<int32_t>(
              {
                  {
                      {3, 3, 6},
                      {2, 1, 5},
                  },
                  {
                      {1, 3, 2},
                      {4, 2, 1},
                  },
              },
              cpu_allocator);

      Kernels::Reverse::cpu_forward_kernel(input, result, attrs);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result=", format_accessor_w_contents(result)));
    }

    SUBCASE("axis = ff_dim_t{1}") {
      ReverseAttrs attrs = ReverseAttrs{
          /*axis=*/ff_dim_t{1_n},
      };

      GenericTensorAccessorR correct =
          create_3d_accessor_r_with_contents<int32_t>(
              {
                  {
                      {4, 2, 1},
                      {1, 3, 2},
                  },
                  {
                      {2, 1, 5},
                      {3, 3, 6},
                  },
              },
              cpu_allocator);

      Kernels::Reverse::cpu_forward_kernel(input, result, attrs);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("axis = ff_dim_t{2}") {
      ReverseAttrs attrs = ReverseAttrs{
          /*axis=*/ff_dim_t{2_n},
      };

      GenericTensorAccessorR correct =
          create_3d_accessor_r_with_contents<int32_t>(
              {
                  {
                      {2, 3, 1},
                      {1, 2, 4},
                  },
                  {
                      {6, 3, 3},
                      {5, 1, 2},
                  },
              },
              cpu_allocator);

      Kernels::Reverse::cpu_forward_kernel(input, result, attrs);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }
  }

  TEST_CASE("Reverse::cpu_backward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_3d_accessor_r_with_contents<int32_t>(
        {
            {
                {1, 3, 2},
                {4, 2, 1},
            },
            {
                {3, 3, 6},
                {2, 1, 5},
            },
        },
        cpu_allocator);

    GenericTensorAccessorW result = create_zero_filled_accessor_w(
        TensorShape{
            TensorDims{FFOrdered{2_p, 2_p, 3_p}},
            DataType::INT32,
        },
        cpu_allocator);

    SUBCASE("axis = ff_dim_t{0}") {
      ReverseAttrs attrs = ReverseAttrs{
          /*axis=*/ff_dim_t{0_n},
      };

      GenericTensorAccessorR correct =
          create_3d_accessor_r_with_contents<int32_t>(
              {
                  {
                      {3, 3, 6},
                      {2, 1, 5},
                  },
                  {
                      {1, 3, 2},
                      {4, 2, 1},
                  },
              },
              cpu_allocator);

      Kernels::Reverse::cpu_forward_kernel(input, result, attrs);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("axis = ff_dim_t{1}") {
      ReverseAttrs attrs = ReverseAttrs{
          /*axis=*/ff_dim_t{1_n},
      };

      GenericTensorAccessorR correct =
          create_3d_accessor_r_with_contents<int32_t>(
              {
                  {
                      {4, 2, 1},
                      {1, 3, 2},
                  },
                  {
                      {2, 1, 5},
                      {3, 3, 6},
                  },
              },
              cpu_allocator);

      Kernels::Reverse::cpu_forward_kernel(input, result, attrs);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("axis = ff_dim_t{2}") {
      ReverseAttrs attrs = ReverseAttrs{
          /*axis=*/ff_dim_t{2_n},
      };

      GenericTensorAccessorR correct =
          create_3d_accessor_r_with_contents<int32_t>(
              {
                  {
                      {2, 3, 1},
                      {1, 2, 4},
                  },
                  {
                      {6, 3, 3},
                      {5, 1, 2},
                  },
              },
              cpu_allocator);

      Kernels::Reverse::cpu_forward_kernel(input, result, attrs);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }
  }
}
