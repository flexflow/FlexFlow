#include "doctest/doctest.h"
#include "kernels/combine_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test combine kernel") {
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<size_t>{100, 100},
        },
        DataType::FLOAT,
    };

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    Allocator allocator = get_local_memory_allocator();

    SUBCASE("Test combine kernel forward") {
      GenericTensorAccessorR accessorR =
          makeReadOnlyAccessor(allocator.allocate_tensor(input_shape));
      GenericTensorAccessorW accessorW = allocator.allocate_tensor(input_shape);

      Kernels::Combine::forward_kernel(stream, accessorR, accessorW);

      std::vector<float> host_output_data(100 * 100);
      checkCUDA(cudaMemcpy(host_output_data.data(),
                           accessorW.ptr,
                           host_output_data.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
    }

    SUBCASE("Test combine kernel backward") {
      GenericTensorAccessorR accessorRGrad =
          makeReadOnlyAccessor(allocator.allocate_tensor(input_shape));
      GenericTensorAccessorW accessorWGrad =
          allocator.allocate_tensor(input_shape);

      Kernels::Combine::backward_kernel(stream, accessorRGrad, accessorWGrad);

      std::vector<float> host_input_grad(100 * 100);
      checkCUDA(cudaMemcpy(host_input_grad.data(),
                           accessorWGrad.ptr,
                           host_input_grad.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
