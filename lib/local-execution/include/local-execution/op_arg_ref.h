#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_ARG_REF_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_ARG_REF_H

#include "local-execution/arg_ref.h"
#include "local-execution/device_specific.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

enum class OpArgRefLabel { PER_DEVICE_OP_STATE, PARALLEL_TENSOR_SHAPE };

struct IndexlessOpArgRefType {
  OpArgRefLabel op_arg_ref_type;
};

struct IndexOpArgRefType {
  OpArgRefLabel op_arg_ref_type;
  int idx;
};

using OpArgRefType = std::variant<IndexlessOpArgRefType, IndexOpArgRefType>;

template <typename T>
using OpArgRef = ArgRef<OpArgRefType, T>;

using OpArgRefSpec = ArgRefSpec<OpArgRefType>;

template <typename T>
OpArgRef<DeviceSpecific<T>> per_device_op_state() {
  OpArgRefType op_arg_ref_type =
      IndexlessOpArgRefType{OpArgRefLabel::PER_DEVICE_OP_STATE};
  ArgRef<OpArgRefType, DeviceSpecific<T>> arg_ref = {op_arg_ref_type};
  return arg_ref;
}

OpArgRef<ParallelTensorShape> input_parallel_tensor_shape(int idx);

} // namespace FlexFlow

#endif
