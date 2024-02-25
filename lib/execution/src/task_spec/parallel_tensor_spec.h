#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_SPEC_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_SPEC_H

#include "pcg/parallel_tensor_guid_t.h"

namespace FlexFlow {

enum class IsGrad { YES, NO };

struct ParallelTensorSpec {
public:
  ParallelTensorSpec grad() const;

public:
  parallel_tensor_guid_t parallel_tensor_guid;
  req<IsGrad> is_grad;
};
FF_VISITABLE_STRUCT(ParallelTensorSpec, parallel_tensor_guid, is_grad);

ParallelTensorSpec grad(parallel_tensor_guid_t const &);

} // namespace FlexFlow

#endif
