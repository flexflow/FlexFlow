#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_MANAGER_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_MANAGER_H

#include "parallel_tensor.h"
#include "tensor.h"

namespace FlexFlow {

struct TensorManager {
  template <typename ...Args>
  Tensor create(Args&&...args) {
    return Tensor(this->tensor_global_guid++, std::forward<Args>(args)...);
  }
private:
  size_t tensor_global_guid = TENSOR_GUID_FIRST_VALID;
};

struct ParallelTensorManager {
  template <typename ...Args>
  ParallelTensor create(Args&&...args) {
    return ParallelTensor(this->parallel_tensor_global_guid++, std::forward<Args>(args)...);
  }
private:
  size_t parallel_tensor_global_guid = PARALLEL_TENSOR_GUID_FIRST_VALID;
};

}

#endif
