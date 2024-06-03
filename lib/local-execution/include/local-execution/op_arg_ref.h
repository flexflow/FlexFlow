#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_ARG_REF_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_ARG_REF_H

#include "local-execution/arg_ref.h"
#include "local-execution/device_specific.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

enum class OpArgRefLabel { PER_DEVICE_OP_STATE, PARALLEL_TENSOR_SHAPE };

struct IndexOpArgRefType {
  OpArgRefLabel op_arg_ref_type;
  int idx = 0;
};

template <typename T>
using OpArgRef = ArgRef<IndexOpArgRefType, T>;

using OpArgRefSpec = ArgRefSpec<IndexOpArgRefType>;

template <typename T>
OpArgRef<DeviceSpecific<T>> per_device_op_state() {
  return {OpArgRefLabel::PER_DEVICE_OP_STATE};
}

OpArgRef<ParallelTensorShape> input_parallel_tensor_shape(int idx);

} // namespace FlexFlow

#endif
