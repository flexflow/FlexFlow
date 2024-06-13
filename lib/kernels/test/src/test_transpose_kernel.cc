#include "doctest/doctest.h"
#include "kernels/transpose_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Transpose Kernel Operations") {
    std::size_t num_elements = 100;
    std::size_t dims[] = {10, 10};
    std::size_t num_dims = 2;
    FlexFlow::ArrayShape shape(dims, num_dims);

    std::vector<ff_dim_t> perm = {ff_dim_t(0), ff_dim_t(1)};

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    float *input_data, *output_data;
    std::vector<float **> ptrs = {&input_data, &output_data};
    std::vector<size_t> sizes = {num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);

    std::vector<float> host_input_data =
        returnRandomFillDeviceData(&input_data, num_elements);
    fillDeviceDataNum(&output_data, num_elements, 0.0f);

    GenericTensorAccessorR input_accessor{DataType::FLOAT, shape, input_data};
    GenericTensorAccessorW output_accessor{DataType::FLOAT, shape, output_data};

    TransposePerDeviceState state =
        Kernels::Transpose::init_kernel(num_dims, perm);

    SUBCASE("Test Transpose Forward Kernel") {
      Kernels::Transpose::forward_kernel(
          stream, state, input_accessor, output_accessor);

      std::vector<float> host_output_data(num_elements);
      checkCUDA(cudaMemcpy(host_output_data.data(),
                           output_data,
                           num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));
    }

    SUBCASE("Test Transpose Backward Kernel") {
      std::vector<float **> grad_ptrs = {&output_data, &input_data};
      allocate_ptrs(grad_ptrs, sizes, allocator);

      Kernels::Transpose::backward_kernel(
          stream, state, output_accessor, input_accessor);

      std::vector<float> host_grad_input_data(num_elements);
      checkCUDA(cudaMemcpy(host_grad_input_data.data(),
                           input_data,
                           num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));
    }

    cudaStreamDestroy(stream);
  }
}
