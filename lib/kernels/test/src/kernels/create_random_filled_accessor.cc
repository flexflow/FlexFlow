#include "kernels/create_random_filled_accessor.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/local_cpu_allocator.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("create_random_filled_accessor_r") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TensorShape shape = TensorShape{
        TensorDims{
            FFOrdered{
                3_p,
                2_p,
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    GenericTensorAccessorR t1 =
        create_random_filled_accessor_r(shape, cpu_allocator, 1);

    SUBCASE("tensors with the same seed are equal") {
      GenericTensorAccessorR t2 =
          create_random_filled_accessor_r(shape, cpu_allocator, 1);

      CHECK(accessors_are_equal(t1, t2));
    }

    SUBCASE("tensors with different seeds are not equal") {
      GenericTensorAccessorR t2 =
          create_random_filled_accessor_r(shape, cpu_allocator, 2);

      CHECK_FALSE(accessors_are_equal(t1, t2));
    }
  }
}
