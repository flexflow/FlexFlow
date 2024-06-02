#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/transpose_kernels.h"
#include "test_utils.h"
#include <algorithm>
#include <iostream>
#include <vector>

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
  TEST_CASE("Test Transpose Forward Kernel") {
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

    const GenericTensorAccessorR input_accessor{DataType::FLOAT, shape,
                                                input_data};
    const GenericTensorAccessorW output_accessor{DataType::FLOAT, shape,
                                                 output_data};

    TransposePerDeviceState state =
        Kernels::Transpose::init_kernel(num_dims, perm);

    Kernels::Transpose::forward_kernel(stream, state, input_accessor,
                                       output_accessor);
    
    checkCUDA(cudaStreamDestroy(stream));
  }

  TEST_CASE("Test Transpose Backward Kernel") {
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

    float *out_grad_data, *in_grad_data;
    std::vector<float **> ptrs = {&out_grad_data, &in_grad_data};
    std::vector<size_t> sizes = {num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);

    std::vector<float> host_out_grad_data =
        returnRandomFillDeviceData(&out_grad_data, num_elements);
    fillDeviceDataNum(&in_grad_data, num_elements, 0.0f);

    const GenericTensorAccessorR out_grad_accessor{DataType::FLOAT, shape,
                                                   out_grad_data};
    const GenericTensorAccessorW in_grad_accessor{DataType::FLOAT, shape,
                                                  in_grad_data};

    TransposePerDeviceState state =
        Kernels::Transpose::init_kernel(num_dims, perm);

    Kernels::Transpose::backward_kernel(stream, state, in_grad_accessor,
                                        out_grad_accessor);

    checkCUDA(cudaStreamDestroy(stream));
  }
}
