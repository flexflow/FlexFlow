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

    std::vector<float> check_output_data(num_elements);
    checkCUDA(cudaMemcpy(check_output_data.data(), output_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<int> in_strides(num_dims, 1);
    std::vector<int> out_strides(num_dims, 1);
    for (int i = 1; i < num_dims; i++) {
      in_strides[i] = in_strides[i - 1] * (shape[legion_dim_t(i)] + 1);
      out_strides[i] = out_strides[i - 1] * (shape[legion_dim_t(perm[i])] + 1);
    }

    std::vector<int> perm_vec(num_dims);
    for (int i = 0; i < num_dims; i++) {
      perm_vec[i] = i;
    }

    for (int o_idx = 0; o_idx < num_elements; ++o_idx) {
      int i_index = 0;
      int t = o_idx;

      for (int i = num_dims - 1; i >= 0; --i) {
        int ratio = t / out_strides[i];
        t -= ratio * out_strides[i];
        i_index += ratio * in_strides[perm_vec[i]];
      }

      CHECK(doctest::Approx(host_input_data[i_index]) ==
            check_output_data[o_idx]);
    }

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

    std::vector<float> check_in_grad_data(num_elements);
    checkCUDA(cudaMemcpy(check_in_grad_data.data(), in_grad_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<int> in_strides(num_dims, 1);
    std::vector<int> out_strides(num_dims, 1);
    for (int i = 1; i < num_dims; i++) {
      in_strides[i] = in_strides[i - 1] * (shape[legion_dim_t(i)] + 1);
      out_strides[i] = out_strides[i - 1] * (shape[legion_dim_t(perm[i])] + 1);
    }

    std::vector<int> perm_vec(num_dims);
    for (int i = 0; i < num_dims; i++) {
      perm_vec[state.perm[i]] = i;
    }

    for (int i_idx = 0; i_idx < num_elements; ++i_idx) {
      int o_idx = 0;
      int t = i_idx;

      for (int i = num_dims - 1; i >= 0; --i) {
        int ratio = t / in_strides[i];
        t -= ratio * in_strides[i];
        o_idx += ratio * out_strides[perm_vec[i]];
      }

      CHECK(doctest::Approx(host_out_grad_data[i_idx]) ==
            check_in_grad_data[o_idx]);
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
