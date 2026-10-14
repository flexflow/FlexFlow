#include "kernels/tensor_accessor_split.h"
#include "kernels/accessor.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("tensor_accessor_split") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {1, 3, 2, 12, -1, 2},
            {2, 1, 5, 9, 5, 10},
        },
        cpu_allocator);

    std::vector<GenericTensorAccessorW> result = tensor_accessor_split(
        input, ff_dim_t{1_n}, {3_p, 1_p, 2_p}, cpu_allocator);

    GenericTensorAccessorR correct_0 =
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 3, 2},
                {2, 1, 5},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_1 =
        create_2d_accessor_r_with_contents<float>(
            {
                {12},
                {9},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_2 =
        create_2d_accessor_r_with_contents<float>(
            {
                {-1, 2},
                {5, 10},
            },
            cpu_allocator);

    CHECK(result.size() == 3);
    CHECK_MESSAGE(
        accessors_are_equal(result.at(0), correct_0),
        check_kv("result.at(0)", format_accessor_w_contents(result.at(0))));
    CHECK_MESSAGE(
        accessors_are_equal(result.at(1), correct_1),
        check_kv("result.at(1)", format_accessor_w_contents(result.at(1))));
    CHECK_MESSAGE(
        accessors_are_equal(result.at(2), correct_2),
        check_kv("result.at(2)", format_accessor_w_contents(result.at(2))));
  }
}
