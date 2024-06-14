#include "doctest/doctest.h"
#include "kernels/cast_kernels.h"
#include "test_utils.h"
#include <type_traits>

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test cast kernel") {
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<size_t>{100, 100},
        },
        DataType::FLOAT,
    };

    Allocator allocator = get_local_memory_allocator();

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    SUBCASE("Test float to double") {
      GenericTensorAccessorR accessorR =
          makeReadOnlyAccessor(allocator.allocate_tensor(input_shape));
      GenericTensorAccessorW accessorW = allocator.allocate_tensor(input_shape);

      Kernels::Cast::forward_kernel(
          nullptr, accessorR, accessorW, DataType::FLOAT, DataType::DOUBLE);

      std::vector<float> host_float_data(100 * 100);
      std::vector<double> host_double_data(100 * 100);

      checkCUDA(cudaMemcpy(host_float_data.data(),
                           accessorR.ptr,
                           host_float_data.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
      checkCUDA(cudaMemcpy(host_double_data.data(),
                           accessorW.ptr,
                           host_double_data.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
    }

    SUBCASE("Test int to float") {
      GenericTensorAccessorR accessorR =
          makeReadOnlyAccessor(allocator.allocate_tensor(input_shape));
      GenericTensorAccessorW accessorW = allocator.allocate_tensor(input_shape);

      Kernels::Cast::forward_kernel(
          nullptr, accessorR, accessorW, DataType::INT32, DataType::FLOAT);

      std::vector<int> host_int_data(100 * 100);
      std::vector<float> host_float_data(100 * 100);

      checkCUDA(cudaMemcpy(host_int_data.data(),
                           accessorR.ptr,
                           host_int_data.size() * sizeof(int),
                           cudaMemcpyDeviceToHost));
      checkCUDA(cudaMemcpy(host_float_data.data(),
                           accessorW.ptr,
                           host_float_data.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
