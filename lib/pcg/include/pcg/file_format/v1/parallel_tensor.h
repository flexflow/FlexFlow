#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_TENSOR_H

#include "utils/json.h"
#include "utils/variant.h"
#include "utils/visitable.h"
#include "initializer.h"
#include "data_type.h"
#include "param_sync.h"

namespace FlexFlow {

struct V1ParallelDim : public use_visitable_cmp<V1ParallelDim> {
  size_t size;
  int degree;
  bool is_replica_dim;
};

struct V1ParallelTensorShape : public use_visitable_cmp<V1ParallelTensorShape> {
  std::vector<V1ParallelDim> dims;
  V1DataType data_type;
};

struct V1ParallelTensor : public use_visitable_cmp<V1ParallelTensor> {
  V1ParallelTensorShape shape;
  optional<V1ParamSync> sync_type;
  optional<V1Initializer> initializer;
  bool create_grad;
};

}

VISITABLE_STRUCT(::FlexFlow::V1ParallelDim, size, degree, is_replica_dim);
MAKE_VISIT_HASHABLE(::FlexFlow::V1ParallelDim);

VISITABLE_STRUCT(::FlexFlow::V1ParallelTensorShape, dims, data_type);
MAKE_VISIT_HASHABLE(::FlexFlow::V1ParallelTensorShape);

VISITABLE_STRUCT(::FlexFlow::V1ParallelTensor, shape, sync_type, initializer, create_grad);
MAKE_VISIT_HASHABLE(::FlexFlow::V1ParallelTensor);

namespace FlexFlow {
static_assert(is_jsonable<V1ParallelDim>::value, "");
static_assert(is_jsonable<V1ParallelTensorShape>::value, "");
static_assert(is_jsonable<V1ParallelTensor>::value, "");
}

#endif
