#ifndef _FLEXFLOW_RUNTIME_SRC_TENSOR_MAPPING_H
#define _FLEXFLOW_RUNTIME_SRC_TENSOR_MAPPING_H

#include "parallel_tensor.h"
#include "tensor.h"

namespace FlexFlow {

struct TensorMapping {
public:
  TensorMapping() = default;

  parallel_tensor_guid_t at(tensor_guid_t) const;
  void add_dependence(tensor_guid_t, parallel_tensor_guid_t);

private:
  std::unordered_map<tensor_guid_t, parallel_tensor_guid_t> contents;
};

} // namespace FlexFlow

#endif
