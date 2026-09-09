#include "kernels/reshape_kernels_cpu.h"
#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reshape_cpu_forward_kernel") {
    Allocator allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {1, 2, 3},
            {4, 5, 6},
        },
        allocator);

    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{3_p, 2_p}},
        DataType::FLOAT,
    };

    // Intentionally randomize this tensor so we can be confident we never read
    // it
    GenericTensorAccessorW output =
        create_random_filled_accessor_w(output_shape, allocator);

    reshape_cpu_forward_kernel(
        /*input=*/input,
        /*output=*/output);

    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<float>(
        {
            {1, 2},
            {3, 4},
            {5, 6},
        },
        allocator);

    CHECK_MESSAGE(accessors_are_equal(output, correct),
                  check_kv("output", format_accessor_w_contents(output)));
  }

  TEST_CASE("reshape_cpu_backward_kernel") {
    Allocator allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {1, 2, 3},
            {4, 5, 6},
        },
        allocator);

    GenericTensorAccessorR output = create_2d_accessor_r_with_contents<float>(
        {
            {1, 2},
            {3, 4},
            {5, 6},
        },
        allocator);

    GenericTensorAccessorR output_grad =
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 2},
                {-3, 4},
                {0.5, 0.25},
            },
            allocator);

    // input_grad is accumulated into, so it needs to start from a known value
    GenericTensorAccessorW input_grad =
        create_2d_accessor_w_with_contents<float>(
            {
                {10, 20, 30},
                {40, 50, 60},
            },
            allocator);

    reshape_cpu_backward_kernel(
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad);

    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<float>(
        {
            {11, 22, 27},
            {44, 50.5, 60.25},
        },
        allocator);

    CHECK_MESSAGE(
        accessors_are_equal(input_grad, correct),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));
  }
}
