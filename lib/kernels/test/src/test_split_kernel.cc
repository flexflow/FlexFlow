#include "doctest/doctest.h"
#include "kernels/split_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Split Forward and Backward Kernel") {
    size_t num_outputs = 2;
    coord_t out_blk_sizes[] = {50, 50};
    coord_t in_blk_size = 100;
    coord_t num_blks = 1;

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape = make_float_tensor_shape_from_legion_dims({100});
    TensorShape output_shape = make_float_tensor_shape_from_legion_dims({50});

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW input_accessor =
          create_random_filled_accessor_w(input_shape, allocator);

      std::vector<float *> output_ptrs = repeat(num_outputs, [&]() {
        GenericTensorAccessorW output_accessor =
            allocator.allocate_tensor(output_shape);
        return output_accessor.get_float_ptr();
      });

      Kernels::Split::forward_kernel(managed_stream.raw_stream(),
                                     output_ptrs.data(),
                                     input_accessor.get_float_ptr(),
                                     out_blk_sizes,
                                     in_blk_size,
                                     num_blks,
                                     num_outputs);
    }

    SUBCASE("backward_kernel") {
      std::vector<float *> output_grad_ptrs(num_outputs);
      for (int i = 0; i < num_outputs; i++) {
        GenericTensorAccessorW output_grad_accessor =
            create_random_filled_accessor_w(output_shape, allocator);
        output_grad_ptrs[i] = output_grad_accessor.get_float_ptr();
      }

      GenericTensorAccessorW input_grad_accessor =
          create_filled_accessor_w(input_shape, allocator, 0.0f);

      Kernels::Split::backward_kernel(managed_stream.raw_stream(),
                                      input_grad_accessor.get_float_ptr(),
                                      (float const **)output_grad_ptrs.data(),
                                      out_blk_sizes,
                                      in_blk_size,
                                      num_blks,
                                      num_outputs);
    }
  }
}
