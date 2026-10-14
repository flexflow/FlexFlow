#include "kernels/accessor_contains_non_zero_value.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/gather_kernels_gpu.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test Gather Forward and Backward Kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    SUBCASE("gpu_forward_kernel") {
      auto run_forward_test = [&](TensorShape input_shape,
                                  TensorShape index_shape,
                                  TensorShape output_shape) {
        ff_dim_t dim = ff_dim_t{
            nonnegative_int{
                get_num_dims(input_shape.dims).int_from_num_tensor_dims() - 1},
        };
        GatherPerDeviceState state =
            Kernels::Gather::gpu_init_kernel(managed_handle.raw_handle(), dim);

        GenericTensorAccessorR input_accessor =
            create_random_filled_accessor_r(input_shape, allocator);
        GenericTensorAccessorR index_accessor =
            create_random_filled_accessor_r(index_shape, allocator);
        GenericTensorAccessorW output_accessor =
            allocator.allocate_tensor(output_shape);

        Kernels::Gather::gpu_forward_kernel(managed_stream.raw_stream(),
                                            state,
                                            input_accessor,
                                            index_accessor,
                                            output_accessor);

        CHECK(contains_non_zero(output_accessor));
      };

      SUBCASE("test gather forward, 2D") {
        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered{2_p, 100_p}},
            DataType::FLOAT,
        };
        TensorShape index_shape = TensorShape{
            TensorDims{FFOrdered{2_p, 20_p}},
            DataType::INT32,
        };
        TensorShape output_shape = TensorShape{
            TensorDims{FFOrdered{2_p, 20_p}},
            DataType::FLOAT,
        };
        run_forward_test(input_shape, index_shape, output_shape);
      }

      SUBCASE("test gather forward, 1D") {
        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered{100_p}},
            DataType::FLOAT,
        };
        TensorShape index_shape = TensorShape{
            TensorDims{FFOrdered{10_p}},
            DataType::INT32,
        };
        TensorShape output_shape = TensorShape{
            TensorDims{FFOrdered{10_p}},
            DataType::FLOAT,
        };
        run_forward_test(input_shape, index_shape, output_shape);
      }
    }

    SUBCASE("gpu_backward_kernel") {
      auto run_backward_test = [&](TensorShape input_shape,
                                   TensorShape index_shape,
                                   TensorShape output_shape) {
        ff_dim_t dim = ff_dim_t{
            nonnegative_int{
                get_num_dims(input_shape.dims).int_from_num_tensor_dims() - 1},
        };
        GatherPerDeviceState state =
            Kernels::Gather::gpu_init_kernel(managed_handle.raw_handle(), dim);

        GenericTensorAccessorR output_grad_accessor =
            create_random_filled_accessor_r(output_shape, allocator);
        GenericTensorAccessorR index_accessor =
            create_random_filled_accessor_r(index_shape, allocator);
        GenericTensorAccessorW input_grad_accessor =
            allocator.allocate_tensor(input_shape);

        Kernels::Gather::gpu_backward_kernel(managed_stream.raw_stream(),
                                             state,
                                             output_grad_accessor,
                                             index_accessor,
                                             input_grad_accessor);
        CHECK(contains_non_zero(input_grad_accessor));
      };

      SUBCASE("test gather backward, 2D") {
        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered{2_p, 100_p}},
            DataType::FLOAT,
        };
        TensorShape index_shape = TensorShape{
            TensorDims{FFOrdered{2_p, 25_p}},
            DataType::INT32,
        };
        TensorShape output_shape = TensorShape{
            TensorDims{FFOrdered{2_p, 25_p}},
            DataType::FLOAT,
        };
        run_backward_test(input_shape, index_shape, output_shape);
      }
    }
  }
}
