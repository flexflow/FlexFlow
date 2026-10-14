#include "kernels/accessor_contains_non_zero_value.h"
#include "kernels/concat_kernels_gpu.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "utils/containers/repeat.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test concat kernel forward and backward") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};
    Allocator allocator = create_local_cuda_memory_allocator();

    const positive_int num_inputs = 4_p;

    SUBCASE("concat_gpu_forward_kernel") {
      auto run_forward_test = [&](positive_int input_rows,
                                  positive_int input_cols,
                                  TensorShape output_shape,
                                  ff_dim_t concat_axis) {
        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered{input_rows, input_cols}},
            DataType::FLOAT,
        };

        std::vector<GenericTensorAccessorR> input_accessors =
            repeat(num_inputs.nonnegative_int_from_positive_int(), [&]() {
              return create_random_filled_accessor_r(input_shape, allocator);
            });

        GenericTensorAccessorW output_accessor =
            allocator.allocate_tensor(output_shape);

        concat_gpu_forward_kernel(managed_stream.raw_stream(),
                                  output_accessor,
                                  input_accessors,
                                  concat_axis);

        CHECK(contains_non_zero(output_accessor));
      };

      SUBCASE("test forward concat, axis = 0") {
        positive_int input_rows = 2_p;
        positive_int input_cols = 4_p;
        TensorShape output_shape = TensorShape{
            TensorDims{FFOrdered{num_inputs * input_rows, input_cols}},
            DataType::FLOAT,
        };
        run_forward_test(input_rows, input_cols, output_shape, ff_dim_t{0_n});
      }

      SUBCASE("test forward concat, axis = 1") {
        positive_int input_rows = 4_p;
        positive_int input_cols = 2_p;
        TensorShape output_shape = TensorShape{
            TensorDims{FFOrdered{input_rows, num_inputs * input_cols}},
            DataType::FLOAT,
        };
        run_forward_test(input_rows, input_cols, output_shape, ff_dim_t{1_n});
      }
    }

    SUBCASE("concat_gpu_backward_kernel") {
      auto run_backward_test = [&](positive_int input_rows,
                                   positive_int input_cols,
                                   TensorShape output_shape,
                                   ff_dim_t concat_axis) {
        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered{input_rows, input_cols}},
            DataType::FLOAT,
        };

        GenericTensorAccessorR output_grad_accessor =
            create_random_filled_accessor_r(output_shape, allocator);

        std::vector<GenericTensorAccessorW> input_grad_accessors =
            repeat(num_inputs.nonnegative_int_from_positive_int(), [&]() {
              return create_zero_filled_accessor_w(input_shape, allocator);
            });

        concat_gpu_backward_kernel(managed_stream.raw_stream(),
                                   output_grad_accessor,
                                   input_grad_accessors,
                                   concat_axis);

        for (auto &accessor : input_grad_accessors) {
          CHECK(contains_non_zero(accessor));
        }
      };

      SUBCASE("test backward concat, axis = 0") {
        positive_int input_rows = 2_p;
        positive_int input_cols = 4_p;
        TensorShape output_shape = TensorShape{
            TensorDims{FFOrdered{num_inputs * input_rows, input_cols}},
            DataType::FLOAT,
        };
        run_backward_test(input_rows, input_cols, output_shape, ff_dim_t{0_n});
      }

      SUBCASE("test backward concat, axis = 1") {
        positive_int input_rows = 4_p;
        positive_int input_cols = 2_p;
        TensorShape output_shape = TensorShape{
            TensorDims{FFOrdered{input_rows, num_inputs * input_cols}},
            DataType::FLOAT,
        };
        run_backward_test(input_rows, input_cols, output_shape, ff_dim_t{1_n});
      }
    }
  }
}
