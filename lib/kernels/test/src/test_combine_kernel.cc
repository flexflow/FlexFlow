#include "doctest/doctest.h"
#include "kernels/combine_kernels.h"
#include "kernels/local_allocator.h"
#include "test_utils.h"

template <typename T>
void allocate_ptrs(std::vector<T **> &gpu_data_ptrs,
                   const std::vector<size_t> &num_elements,
                   Allocator &allocator) {
  for (size_t i = 0; i < gpu_data_ptrs.size(); ++i) {
    *gpu_data_ptrs[i] =
        static_cast<T *>(allocator.allocate(num_elements[i] * sizeof(float)));
  }
}

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test combine kernel forward") {
    std::size_t dims[] = {100, 100};
    std::size_t num_dims = 2;
    FlexFlow::ArrayShape shape(dims, num_dims);
    std::size_t num_elements = 100 * 100;

    Allocator allocator = get_local_memory_allocator();

    void *input_data_ptr, *output_data_ptr;
    std::vector<void **> ptrs = {&input_data_ptr, &output_data_ptr};
    std::vector<size_t> sizes = {num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);
    std::vector<float> host_input_data =
        returnRandomFillDeviceData(&input_data_ptr, num_elements);

    const GenericTensorAccessorR accessorR{DataType::FLOAT, shape,
                                           input_data_ptr};
    const GenericTensorAccessorW accessorW{DataType::FLOAT, shape,
                                           output_data_ptr};

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Kernels::Combine::forward_kernel(stream, accessorR, accessorW);

    std::vector<float> host_output_data(100 * 100);
    checkCUDA(cudaMemcpy(host_output_data.data(), output_data_ptr,
                         host_output_data.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < num_elements; ++i) {
      REQUIRE(host_output_data[i] == host_input_data[i]);
    }

    checkCUDA(cudaStreamDestroy(stream));
  }

  TEST_CASE("Test combine kernel backward") {
    std::size_t dims[] = {100, 100};
    std::size_t num_dims = 2;
    FlexFlow::ArrayShape shape(dims, num_dims);

    Allocator allocator = get_local_memory_allocator();

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    void *grad_output_data_ptr, *grad_input_data_ptr;
    std::vector<void **> ptrs = {&grad_output_data_ptr, &grad_input_data_ptr};
    std::vector<size_t> sizes = {100 * 100, 100 * 100};
    allocate_ptrs(ptrs, sizes, allocator);
    fillDeviceDataOnes(&grad_output_data_ptr, 100 * 100);
    fillDeviceDataZeros(&grad_input_data_ptr, 100 * 100);

    const GenericTensorAccessorR accessorRGrad{DataType::FLOAT, shape,
                                               grad_output_data_ptr};
    const GenericTensorAccessorW accessorWGrad{DataType::FLOAT, shape,
                                               grad_input_data_ptr};

    Kernels::Combine::backward_kernel(stream, accessorRGrad, accessorWGrad);

    std::vector<float> host_input_grad(100 * 100);
    checkCUDA(cudaMemcpy(host_input_grad.data(), grad_input_data_ptr,
                         host_input_grad.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (float val : host_input_grad) {
      REQUIRE(val == 1.0f);
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
