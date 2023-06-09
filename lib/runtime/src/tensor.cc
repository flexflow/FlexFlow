#include "tensor.h"
#include "parallel_tensor.h"
#include "utils/optional.h"

namespace FlexFlow {

TensorBase::TensorBase(TensorBase const &rhs) {
  tensor_guid = rhs.tensor_guid;
  num_dims = rhs.num_dims;
  for (int i = 0; i < num_dims; i++) {
    dims[i] = rhs.dims[i];
  }
  data_type = rhs.data_type;
  sync_type = rhs.sync_type;
  initializer = rhs.initializer;
  parallel_tensor = rhs.parallel_tensor;
  owner_layer = rhs.owner_layer;
  owner_idx = rhs.owner_idx;
  create_gradients = rhs.create_gradients;
}

size_t TensorBase::get_volume() const {
  size_t volume = 1;
  for (int i = 0; i < num_dims; i++) {
    volume *= dims[i];
  }
  return volume;
}

template <typename T>
bool TensorBase::set_tensor(FFModel const *ff,
                            std::vector<int> const &dim_sizes,
                            T const *data) {
  if (num_dims != (int)dim_sizes.size()) {
    return false;
  }
  for (int i = 0; i < num_dims; i++) {
    if (dims[num_dims - 1 - i] != dim_sizes[i]) {
      return false;
    }
  }
  ParallelTensor ptensor = nullptr;
  ff->get_parallel_tensor_from_tensor(this, ptensor);
  ptensor->set_tensor<T>(ff, dim_sizes, data);
  return true;
}

template <typename T>
bool TensorBase::get_tensor(FFModel const *ff, T *data, bool get_gradients) {
  optional<ParallelTensor> ptensor = nullopt;
  ff->get_parallel_tensor_from_tensor(this, ptensor);
  ptensor->get_tensor<T>(ff, data, get_gradients);
  return true;
}

template <typename T>
bool TensorBase::get_output_parallel_tensor(FFModel const *ff,
                                            T *data,
                                            bool get_gradients) {
  ParallelTensor parallel_tensor = nullptr;
  Op *final_operator = ff->get_final_operator();
  assert(final_operator->numOutputs == 1);
  parallel_tensor = final_operator->outputs[0];
  parallel_tensor->get_tensor<T>(ff, data, get_gradients);
  return true;
}

} // namespace FlexFlow
