#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_SPEC_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_SPEC_H

#include "parallel_tensor_guid_t.h"

namespace FlexFlow {

enum class IsGrad { YES, NO };

struct ParallelTensorSpec : public use_visitable_cmp<ParallelTensorSpec> {
public:
  ParallelTensorSpec() = delete;
  ParallelTensorSpec(parallel_tensor_guid_t, IsGrad is_grad = IsGrad::NO);

  ParallelTensorSpec grad() const;

public:
  parallel_tensor_guid_t parallel_tensor_guid;
  IsGrad is_grad;
};

ParallelTensorSpec grad(parallel_tensor_guid_t const &);

} // namespace FlexFlow

#endif
