#include "test_utils.h"

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
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

GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
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

void fill_tensor_accessor_w(GenericTensorAccessorW accessor, float val) {
  LegionTensorDims dims = accessor.shape.dims;

  int volume =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  std::vector<float> host_data(volume, val);
  checkCUDA(cudaMemcpy(accessor.ptr,
                       host_data.data(),
                       host_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
}

TensorShape make_float_tensor_shape_w_legion_dims(FFOrdered<size_t> dims) {
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

void setPerDeviceFFHandle(PerDeviceFFHandle *handle) {
  cudnnCreate(&handle->dnn);
  cublasCreate(&handle->blas);
  handle->workSpaceSize = 1024 * 1024;
  cudaMalloc(&handle->workSpace, handle->workSpaceSize);
  handle->allowTensorOpMathConversion = true;
}

PerDeviceFFHandle get_per_device_ff_handle() {
  PerDeviceFFHandle handle;
  setPerDeviceFFHandle(&handle);
  return handle;
}

void cleanup_test(cudaStream_t &stream, PerDeviceFFHandle &handle) {
  checkCUDA(cudaStreamDestroy(stream));
  checkCUDA(cudaFree(handle.workSpace));
  cudnnDestroy(handle.dnn);
  cublasDestroy(handle.blas);
}
