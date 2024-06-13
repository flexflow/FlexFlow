#include "doctest/doctest.h"
#include "kernels/combine_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test combine kernel") {
    ArrayShape shape = ArrayShape{
        std::vector<size_t>{100, 100},
    };
    std::size_t num_elements = 100 * 100;

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    Allocator allocator = get_local_memory_allocator();

    SUBCASE("Test combine kernel forward") {
      void *input_data_ptr, *output_data_ptr;
      std::vector<void **> ptrs = {&input_data_ptr, &output_data_ptr};
      std::vector<size_t> sizes = {num_elements, num_elements};
      allocate_ptrs(ptrs, sizes, allocator);
      std::vector<float> host_input_data =
          returnRandomFillDeviceData(&input_data_ptr, num_elements);

      GenericTensorAccessorR accessorR{DataType::FLOAT, shape, input_data_ptr};
      GenericTensorAccessorW accessorW{DataType::FLOAT, shape, output_data_ptr};

      Kernels::Combine::forward_kernel(stream, accessorR, accessorW);

      std::vector<float> host_output_data(100 * 100);
      checkCUDA(cudaMemcpy(host_output_data.data(),
                           output_data_ptr,
                           host_output_data.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));

      for (size_t i = 0; i < num_elements; ++i) {
        REQUIRE(host_output_data[i] == host_input_data[i]);
      }
    }

    SUBCASE("Test combine kernel backward") {
      void *grad_output_data_ptr, *grad_input_data_ptr;
      std::vector<void **> ptrs = {&grad_output_data_ptr, &grad_input_data_ptr};
      std::vector<size_t> sizes = {100 * 100, 100 * 100};
      allocate_ptrs(ptrs, sizes, allocator);
      fillDeviceDataOnes(&grad_output_data_ptr, 100 * 100);
      fillDeviceDataZeros(&grad_input_data_ptr, 100 * 100);

      GenericTensorAccessorR accessorRGrad{
          DataType::FLOAT, shape, grad_output_data_ptr};
      GenericTensorAccessorW accessorWGrad{
          DataType::FLOAT, shape, grad_input_data_ptr};

      Kernels::Combine::backward_kernel(stream, accessorRGrad, accessorWGrad);

      std::vector<float> host_input_grad(100 * 100);
      checkCUDA(cudaMemcpy(host_input_grad.data(),
                           grad_input_data_ptr,
                           host_input_grad.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));

      for (float val : host_input_grad) {
        REQUIRE(val == 1.0f);
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
