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

    ffStream_t stream = create_ff_stream();

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
      Kernels::Split::forward_kernel(stream,
                                     output_ptrs.data(),
                                     input_accessor.get_float_ptr(),
                                     out_blk_sizes,
                                     in_blk_size,
                                     num_blks,
                                     num_outputs);

      std::vector<std::vector<float>> host_output_data(
          num_outputs, std::vector<float>(50, 0));
      for (int i = 0; i < num_outputs; i++) {
        host_output_data[i] =
            load_vector_to_host_from_device(output_ptrs[i], out_blk_sizes[i]);
      }

      // Will add this back once CPU tests are finished
      // for (int i = 0; i < num_outputs; i++) {
      //   int offset = std::accumulate(out_blk_sizes, out_blk_sizes + i, 0);
      //   for (int j = 0; j < out_blk_sizes[i]; j++) {
      //     CHECK(host_output_data[i][j] == host_input_data[offset + j]);
      //   }
      // }

      SUBCASE("backward_kernel") {
        float *grad_input_data = static_cast<float *>(allocator.allocate(
            input_accessor.shape.num_elements() * sizeof(float)));
        cudaMemset(grad_input_data,
                   0,
                   input_accessor.shape.num_elements() * sizeof(float));

        Kernels::Split::backward_kernel(stream,
                                        grad_input_data,
                                        (float const **)(output_ptrs.data()),
                                        out_blk_sizes,
                                        in_blk_size,
                                        num_blks,
                                        num_outputs);

        // Will add this back once CPU tests are finished
        // std::vector<float> host_grad_input_data(
        //     input_accessor.shape.num_elements(), 0);
        // cudaMemcpy(host_grad_input_data.data(),
        //            grad_input_data,
        //            input_accessor.shape.num_elements() * sizeof(float),
        //            cudaMemcpyDeviceToHost);
      }
    }

    cudaStreamDestroy(stream);
  }
}
