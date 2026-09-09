#include "kernels/batch_matmul_kernels_gpu.h"
#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static GenericTensorAccessorR make_input_lhs(Allocator &allocator) {
  return create_3d_accessor_r_with_contents<float>(
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
      allocator);
}

static GenericTensorAccessorR make_input_rhs(Allocator &allocator) {
  return create_3d_accessor_r_with_contents<float>(
      {
          {
              {1.0, 0.5},
              {2.0, 4.0},
              {1.5, -1.0},
          },
          {
              {-6.0, -0.5},
              {1.0, -2.0},
              {0.4, -3.5},
          },
      },
      allocator);
}

static GenericTensorAccessorR make_output(Allocator &allocator) {
  return create_3d_accessor_r_with_contents<float>(
      {
          {
              {18.0, 7.5},
              {11.5, 0.0},
              {2.0, 10.5},
              {4.5, 9.0},
          },
          {
              {-29.799999237060547, 2.5},
              {47.599998474121094, 7.5},
              {-2.5999999046325684, 7.474999904632568},
              {-3.0, 19.5},
          },
      },
      allocator);
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("batch_matmul_gpu_forward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    GenericTensorAccessorR input_lhs = make_input_lhs(allocator);
    GenericTensorAccessorR input_rhs = make_input_rhs(allocator);

    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{2_p, 4_p, 2_p}},
        DataType::FLOAT,
    };

    // Intentionally randomize this tensor so we can be confident we never read
    // it
    GenericTensorAccessorW result =
        create_random_filled_accessor_w(output_shape, allocator);

    batch_matmul_gpu_forward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*handle=*/managed_handle.raw_handle(),
        /*input_lhs=*/input_lhs,
        /*input_rhs=*/input_rhs,
        /*output=*/result);

    GenericTensorAccessorR correct = make_output(allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("batch_matmul_gpu_backward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    GenericTensorAccessorR input_lhs = make_input_lhs(allocator);
    GenericTensorAccessorR input_rhs = make_input_rhs(allocator);
    GenericTensorAccessorR output = make_output(allocator);

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
            allocator);

    // The gradients are overwritten rather than accumulated into (matching
    // batch_matmul_cpu_backward_kernel), so randomize them to be confident
    // their previous contents are never read
    GenericTensorAccessorW input_lhs_grad = create_random_filled_accessor_w(
        get_tensor_shape_for_accessor_r(input_lhs), allocator);

    GenericTensorAccessorW input_rhs_grad = create_random_filled_accessor_w(
        get_tensor_shape_for_accessor_r(input_rhs), allocator);

    batch_matmul_gpu_backward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*handle=*/managed_handle.raw_handle(),
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input_lhs=*/input_lhs,
        /*input_lhs_grad=*/input_lhs_grad,
        /*input_rhs=*/input_rhs,
        /*input_rhs_grad=*/input_rhs_grad);

    GenericTensorAccessorR correct_input_lhs_grad =
        create_3d_accessor_r_with_contents<float>(
            {
                {
                    {2.0, 10.0, -0.5},
                    {-0.4000000059604645,
                     -3.799999952316284,
                     1.149999976158142},
                    {1.25, 7.0, -0.75},
                    {3.0, -3.0, 9.75},
                },
                {
                    {1.25, 5.0, 8.75},
                    {-5.75, 2.0, 2.1500000953674316},
                    {12.25, -1.0, 0.949999988079071},
                    {17.875, -3.5, -2.075000047683716},
                },
            },
            allocator);

    GenericTensorAccessorR correct_input_rhs_grad =
        create_3d_accessor_r_with_contents<float>(
            {
                {
                    {39.70000076293945, -18.5},
                    {6.349999904632568, 6.5},
                    {-8.0, 13.0},
                },
                {
                    {-8.5, -8.625},
                    {3.5999999046325684, -2.5999999046325684},
                    {18.0, 5.25},
                },
            },
            allocator);

    CHECK_MESSAGE(
        accessors_are_equal(input_lhs_grad, correct_input_lhs_grad),
        check_kv("input_lhs_grad", format_accessor_w_contents(input_lhs_grad)));

    CHECK_MESSAGE(
        accessors_are_equal(input_rhs_grad, correct_input_rhs_grad),
        check_kv("input_rhs_grad", format_accessor_w_contents(input_rhs_grad)));
  }
}
