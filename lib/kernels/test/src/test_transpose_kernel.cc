#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/transpose_kernels.h"
#include <algorithm>
#include <iostream>
#include <vector>

namespace FlexFlow {

struct TransposeStrides {
  int num_dim;
  int in_strides[MAX_TENSOR_DIM], out_strides[MAX_TENSOR_DIM],
      perm[MAX_TENSOR_DIM];
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Transpose Forward Kernel") {
    std::size_t num_elements = 100;
    std::size_t dims[] = {10, 10};
    std::size_t num_dims = 2;
    FlexFlow::ArrayShape shape(dims, num_dims);

    std::vector<ff_dim_t> perm = {ff_dim_t(0), ff_dim_t(1)};

    PerDeviceFFHandle handle;
    cudnnCreate(&handle.dnn);
    cublasCreate(&handle.blas);
    handle.workSpaceSize = 1024 * 1024;
    cudaMalloc(&handle.workSpace, handle.workSpaceSize);
    handle.allowTensorOpMathConversion = true;

    TransposePerDeviceState state =
        Kernels::Transpose::init_kernel(num_dims, perm);

    Allocator allocator = get_local_memory_allocator();
    float *input_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    float *output_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));

    std::vector<float> host_input_data(num_elements);
    std::generate(host_input_data.begin(), host_input_data.end(),
                  []() { return static_cast<float>(rand()) / RAND_MAX; });
    checkCUDA(cudaMemcpy(input_data, host_input_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> host_output_data(num_elements, 0.0f);
    checkCUDA(cudaMemcpy(output_data, host_output_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    const GenericTensorAccessorR input_accessor{DataType::FLOAT, shape,
                                                input_data};
    const GenericTensorAccessorW output_accessor{DataType::FLOAT, shape,
                                                 output_data};

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
    cudnnCreate(&handle.dnn);
    cublasCreate(&handle.blas);
    handle.workSpaceSize = 1024 * 1024;
    cudaMalloc(&handle.workSpace, handle.workSpaceSize);
    handle.allowTensorOpMathConversion = true;

    TransposePerDeviceState state =
        Kernels::Transpose::init_kernel(num_dims, perm);

    Allocator allocator = get_local_memory_allocator();
    float *out_grad_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    float *in_grad_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));

    std::vector<float> host_out_grad_data(num_elements);
    std::generate(host_out_grad_data.begin(), host_out_grad_data.end(),
                  []() { return static_cast<float>(rand()) / RAND_MAX; });
    checkCUDA(cudaMemcpy(out_grad_data, host_out_grad_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> host_in_grad_data(num_elements, 0.0f);
    checkCUDA(cudaMemcpy(in_grad_data, host_in_grad_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    const GenericTensorAccessorR out_grad_accessor{DataType::FLOAT, shape,
                                                   out_grad_data};
    const GenericTensorAccessorW in_grad_accessor{DataType::FLOAT, shape,
                                                  in_grad_data};

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
} // namespace FlexFlow
