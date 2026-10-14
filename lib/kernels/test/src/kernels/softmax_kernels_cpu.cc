#include "kernels/softmax_kernels_cpu.h"
#include "kernels/accessors_are_approximately_equal.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("softmax_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {-1, -2, -3},
        },
        cpu_allocator);

    SUBCASE("dim = 0") {
      SoftmaxAttrs attrs{
          /*dim=*/ff_dim_t{0_n},
      };

      // Intentionally randomize this tensor so we can be confident we never read it
      GenericTensorAccessorW output =
          create_random_filled_accessor_w(input.shape, cpu_allocator);

      softmax_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/output);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {0.0023548827, 0.0023555316, 0.0023556193},
                  {0.04729908, 0.047312114, 0.047313876},
                  {0.9500274, 0.95028925, 0.95032465},
                  {0.00031869867, 4.3143067e-05, 5.8389965e-06},
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_approximately_equal(output, correct),
                    check_kv("output", format_accessor_w_contents(output)));
    }

    SUBCASE("dim = 1") {
      SoftmaxAttrs attrs{
          /*dim=*/ff_dim_t{1_n},
      };

      // Intentionally randomize this tensor so we can be confident we never read it
      GenericTensorAccessorW output =
          create_random_filled_accessor_w(input.shape, cpu_allocator);

      softmax_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/output);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {0.090030566, 0.24472848, 0.6652409},
                  {0.090030566, 0.24472848, 0.6652409},
                  {0.090030566, 0.24472848, 0.6652409},
                  {0.6652409, 0.24472848, 0.090030566},
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_approximately_equal(output, correct),
                    check_kv("output", format_accessor_w_contents(output)));
    }
  }
}
