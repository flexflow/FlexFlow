#include "kernels/concat_kernels_cpu.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("concat_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input_0 = create_2d_accessor_r_with_contents<float>(
        {
            {1, 3, 2},
            {2, 1, 5},
        },
        cpu_allocator);

    GenericTensorAccessorR input_1 = create_2d_accessor_r_with_contents<float>(
        {
            {12},
            {9},
        },
        cpu_allocator);

    GenericTensorAccessorR input_2 = create_2d_accessor_r_with_contents<float>(
        {
            {-1, 2},
            {5, 10},
        },
        cpu_allocator);

    GenericTensorAccessorR correct_output =
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 3, 2, 12, -1, 2},
                {2, 1, 5, 9, 5, 10},
            },
            cpu_allocator);

    GenericTensorAccessorW output =
        create_zero_filled_accessor_w(correct_output.shape, cpu_allocator);

    concat_cpu_forward_kernel(
        /*output=*/output,
        /*inputs=*/{input_0, input_1, input_2},
        /*axis=*/ff_dim_t{1_n});

    CHECK_MESSAGE(accessors_are_equal(output, correct_output),
                  check_kv("output", format_accessor_w_contents(output)));
  }

  TEST_CASE("concat_cpu_backward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR output_grad =
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 3, 2, 12, -1, 2},
                {2, 1, 5, 9, 5, 10},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_input_grad_0 =
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 3, 2},
                {2, 1, 5},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_input_grad_1 =
        create_2d_accessor_r_with_contents<float>(
            {
                {12},
                {9},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_input_grad_2 =
        create_2d_accessor_r_with_contents<float>(
            {
                {-1, 2},
                {5, 10},
            },
            cpu_allocator);

    GenericTensorAccessorW input_grad_0 = create_zero_filled_accessor_w(
        correct_input_grad_0.shape, cpu_allocator);

    GenericTensorAccessorW input_grad_1 = create_zero_filled_accessor_w(
        correct_input_grad_1.shape, cpu_allocator);

    GenericTensorAccessorW input_grad_2 = create_zero_filled_accessor_w(
        correct_input_grad_2.shape, cpu_allocator);

    concat_cpu_backward_kernel(
        /*output_grad=*/output_grad,
        /*input_grads=*/{input_grad_0, input_grad_1, input_grad_2},
        /*axis=*/ff_dim_t{1_n});

    CHECK_MESSAGE(
        accessors_are_equal(input_grad_0, correct_input_grad_0),
        check_kv("input_grad_0", format_accessor_w_contents(input_grad_0)));
    CHECK_MESSAGE(
        accessors_are_equal(input_grad_1, correct_input_grad_1),
        check_kv("input_grad_1", format_accessor_w_contents(input_grad_1)));
    CHECK_MESSAGE(
        accessors_are_equal(input_grad_2, correct_input_grad_2),
        check_kv("input_grad_2", format_accessor_w_contents(input_grad_2)));
  }
}
