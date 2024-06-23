#include "doctest/doctest.h"
#include "kernels/split_kernels.h"
#include "test_utils.h"
#include <numeric>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Split Forward and Backward Kernel") {
    size_t num_outputs = 2;
    coord_t out_blk_sizes[] = {50, 50};
    coord_t in_blk_size = 100;
    coord_t num_blks = 1;

    ManagedStream mStream = get_managed_stream();

    Allocator allocator = get_local_memory_allocator();

    TensorShape input_shape = make_float_tensor_shape_from_legion_dims({100});
    GenericTensorAccessorW input_accessor =
        create_random_filled_accessor_w(input_shape, allocator);
    std::vector<float> host_input_data = load_data_to_host_from_device<float>(
        read_only_accessor_from_write_accessor(input_accessor));

    std::vector<float *> output_ptrs(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
      output_ptrs[i] = static_cast<float *>(
          allocator.allocate(out_blk_sizes[i] * sizeof(float)));
    }

    SUBCASE("forward_kernel") {
      Kernels::Split::forward_kernel(mStream.stream,
                                     output_ptrs.data(),
                                     input_accessor.get_float_ptr(),
                                     out_blk_sizes,
                                     in_blk_size,
                                     num_blks,
                                     num_outputs);
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW grad_input_accessor =
          create_filled_accessor_w(input_shape, allocator, 0.0f);

      Kernels::Split::backward_kernel(mStream.stream,
                                      grad_input_accessor.get_float_ptr(),
                                      (float const **)(output_ptrs.data()),
                                      out_blk_sizes,
                                      in_blk_size,
                                      num_blks,
                                      num_outputs);
    }
  }
}
