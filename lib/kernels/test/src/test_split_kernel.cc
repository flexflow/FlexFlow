#include "kernels/create_constant_filled_accessor_r.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "kernels/split_kernels_gpu.h"
#include "op-attrs/datatype_value.h"
#include "utils/containers/repeat.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test Split Forward and Backward Kernel") {
    nonnegative_int num_outputs = 2_n;
    int out_blk_sizes[] = {50, 50};
    int in_blk_size = 100;
    int num_blks = 1;

    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{100_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{50_p}},
        DataType::FLOAT,
    };

    SUBCASE("gpu_forward_kernel") {
      GenericTensorAccessorW input_accessor =
          create_random_filled_accessor_w(input_shape, allocator);

      std::vector<float *> output_ptrs = repeat(num_outputs, [&]() {
        GenericTensorAccessorW output_accessor =
            allocator.allocate_tensor(output_shape);
        return output_accessor.get_float_ptr();
      });

      split_gpu_forward_kernel(managed_stream.raw_stream(),
                               output_ptrs.data(),
                               input_accessor.get_float_ptr(),
                               out_blk_sizes,
                               in_blk_size,
                               num_blks,
                               num_outputs.unwrap_nonnegative());
    }

    SUBCASE("gpu_backward_kernel") {
      std::vector<float *> output_grad_ptrs(num_outputs.unwrap_nonnegative());
      for (int i = 0; i < num_outputs; i++) {
        GenericTensorAccessorW output_grad_accessor =
            create_random_filled_accessor_w(output_shape, allocator);
        output_grad_ptrs[i] = output_grad_accessor.get_float_ptr();
      }

      GenericTensorAccessorW input_grad_accessor =
          create_constant_filled_accessor_w(
              input_shape, allocator, make_float_data_type_value(0));

      split_gpu_backward_kernel(managed_stream.raw_stream(),
                                input_grad_accessor.get_float_ptr(),
                                (float const **)output_grad_ptrs.data(),
                                out_blk_sizes,
                                in_blk_size,
                                num_blks,
                                num_outputs.unwrap_nonnegative());
    }
  }
}
