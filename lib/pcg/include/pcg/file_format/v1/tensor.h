#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_H

#include <vector>
#include "data_type.h"
#include "initializer.h"
#include "utils/visitable.h"
#include "param_sync.h"

namespace FlexFlow {

struct V1TensorShape : public use_visitable_cmp<V1TensorShape> {
  std::vector<size_t> dims;
  V1DataType data_type;
};

struct V1Tensor : public use_visitable_cmp<V1Tensor> {
  V1TensorShape shape;
  optional<V1Initializer> initializer;
  bool create_gradients;
  optional<V1ParamSync> sync_type;
  optional<std::string> name;
};

}

VISITABLE_STRUCT(::FlexFlow::V1TensorShape, dims, data_type);
MAKE_VISIT_HASHABLE(::FlexFlow::V1TensorShape);

VISITABLE_STRUCT(::FlexFlow::V1Tensor, shape, initializer, create_gradients, sync_type, name);
MAKE_VISIT_HASHABLE(::FlexFlow::V1Tensor);

namespace FlexFlow {
static_assert(is_jsonable<V1TensorShape>::value, "");
static_assert(is_jsonable<V1Tensor>::value, "");
}

#endif
