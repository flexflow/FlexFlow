#include "test_utils.h"

GenericTensorAccessorW getRandomFilledAccessorW(TensorShape const &shape,
                                                Allocator &allocator) {
  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
  FFOrdered<size_t> dims = shape.dims.ff_ordered;

  int volume =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  std::vector<float> host_data(volume);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &val : host_data) {
    val = dist(gen);
  }
  checkCUDA(cudaMemcpy(accessor.ptr,
                       host_data.data(),
                       host_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
  return accessor;
}

GenericTensorAccessorW getFilledAccessorW(TensorShape const &shape,
                                          Allocator &allocator,
                                          float val) {
  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
  FFOrdered<size_t> dims = shape.dims.ff_ordered;

  int volume =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  std::vector<float> host_data(volume, val);
  checkCUDA(cudaMemcpy(accessor.ptr,
                       host_data.data(),
                       host_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
  return accessor;
}

TensorShape get_float_tensor_shape(FFOrdered<size_t> dims) {
  return TensorShape{
      TensorDims{
          dims,
      },
      DataType::FLOAT,
  };
}

TensorShape get_double_tensor_shape(FFOrdered<size_t> dims) {
  return TensorShape{
      TensorDims{
          dims,
      },
      DataType::DOUBLE,
  };
}

void cleanup_test(cudaStream_t &stream, PerDeviceFFHandle &handle) {
  checkCUDA(cudaStreamDestroy(stream));
  checkCUDA(cudaFree(handle.workSpace));
  cudnnDestroy(handle.dnn);
  cublasDestroy(handle.blas);
}
