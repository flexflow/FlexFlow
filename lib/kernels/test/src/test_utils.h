#ifndef _FLEXFLOW_KERNELS_TEST_UTILS
#define _FLEXFLOW_KERNELS_TEST_UTILS

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include <algorithm>
#include <memory>
#include <random>
#include <vector>

template <typename T>
void randomFillDeviceData(T **gpu_data, size_t num_elements) {
  std::vector<float> host_data(num_elements);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &val : host_data) {
    val = dist(gen);
  }
  checkCUDA(cudaMemcpy(*gpu_data,
                       host_data.data(),
                       host_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<float> returnRandomFillDeviceData(T **gpu_data,
                                              size_t num_elements) {
  std::vector<float> host_data(num_elements);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &val : host_data) {
    val = dist(gen);
  }
  checkCUDA(cudaMemcpy(*gpu_data,
                       host_data.data(),
                       host_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice));

  return host_data;
}

template <typename T>
void fillDeviceDataNum(T **gpu_data, size_t num_elements, T num) {
  std::vector<float> host_data(num_elements, num);
  checkCUDA(cudaMemcpy(*gpu_data,
                       host_data.data(),
                       host_data.size() * sizeof(T),
                       cudaMemcpyHostToDevice));
}

template <typename T>
void fillDeviceDataIota(T **gpu_data, size_t num_elements) {
  std::vector<float> host_data(num_elements);
  std::iota(host_data.begin(), host_data.end(), 0.0f);
  checkCUDA(cudaMemcpy(*gpu_data,
                       host_data.data(),
                       host_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
}

template <typename T>
void fillDeviceDataOnes(T **gpu_data, size_t num_elements) {
  std::vector<float> host_data(num_elements, 1.0f);
  checkCUDA(cudaMemcpy(*gpu_data,
                       host_data.data(),
                       host_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
}

template <typename T>
void fillDeviceDataZeros(T **gpu_data, size_t num_elements) {
  std::vector<float> host_data(num_elements, 0.0f);
  checkCUDA(cudaMemcpy(*gpu_data,
                       host_data.data(),
                       host_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
}

template <typename T>
void fillDeviceDataPtrsOnes(std::vector<T **> &gpu_data_ptrs,
                            std::vector<size_t> &num_elements) {
  for (int i = 0; i < gpu_data_ptrs.size(); i++) {
    fillDeviceDataOnes(gpu_data_ptrs[i], num_elements[i]);
  }
}

template <typename T>
void fillDeviceDataPtrsZeros(std::vector<T **> &gpu_data_ptrs,
                             std::vector<size_t> &num_elements) {
  for (int i = 0; i < gpu_data_ptrs.size(); i++) {
    fillDeviceDataZeros(gpu_data_ptrs[i], num_elements[i]);
  }
}

template <typename T>
void randomFillDevicePtrs(std::vector<T **> &gpu_data_ptrs,
                          std::vector<size_t> &num_elements) {
  for (int i = 0; i < gpu_data_ptrs.size(); i++) {
    randomFillDeviceData(gpu_data_ptrs[i], num_elements[i]);
  }
}

template <typename T>
inline bool contains_non_zero(std::vector<T> &data) {
  for (auto &val : data) {
    if (val != 0) {
      return true;
    }
  }
  return false;
}

inline void setPerDeviceFFHandle(PerDeviceFFHandle *handle) {
  cudnnCreate(&handle->dnn);
  cublasCreate(&handle->blas);
  handle->workSpaceSize = 1024 * 1024;
  cudaMalloc(&handle->workSpace, handle->workSpaceSize);
  handle->allowTensorOpMathConversion = true;
}
#endif
