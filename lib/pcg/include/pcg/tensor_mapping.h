#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_MAPPING_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_MAPPING_H

#include "tensor.h"
#include "parallel_tensor.h"

namespace FlexFlow {

struct TensorMapping {
public:
  TensorMapping() = default;

  parallel_tensor_guid_t at(tensor_guid_t) const;
  void add_dependence(tensor_guid_t, parallel_tensor_guid_t);
private:
  std::unordered_map<tensor_guid_t, parallel_tensor_guid_t> contents;
};

}

namespace FlexFlow {
static_assert(is_well_behaved_value_type<TensorMapping>::value, "");
}

#endif 
