#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_MAPPING_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_MAPPING_H

#include "parallel_tensor_guid_t.h"
#include "tensor_guid_t.h"

namespace FlexFlow {

struct TensorMapping
    : public strong_typedef<
          TensorMapping,
          std::unordered_map<tensor_guid_t, parallel_tensor_guid_t>> {
public:
  TensorMapping();

  parallel_tensor_guid_t at(tensor_guid_t) const;
  void add_dependence(tensor_guid_t, parallel_tensor_guid_t);

private:
  std::unordered_map<tensor_guid_t, parallel_tensor_guid_t> contents;
};

} // namespace FlexFlow

MAKE_TYPEDEF_PRINTABLE(::FlexFlow::TensorMapping, "TensorMapping");
MAKE_TYPEDEF_HASHABLE(::FlexFlow::TensorMapping);

namespace FlexFlow {
static_assert(is_well_behaved_value_type<TensorMapping>::value, "");
}

#endif
