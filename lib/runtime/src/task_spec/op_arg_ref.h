#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_OP_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_OP_ARG_REF_H

#include "arg_ref.h"
#include "device_specific.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

enum class OpArgRefType { PER_DEVICE_OP_STATE, PARALLEL_TENSOR_SHAPE };

template <typename T>
using OpArgRef = ArgRef<OpArgRefType, T>;

using OpArgRefSpec = ArgRefSpec<OpArgRefType>;

template <typename T>
OpArgRef<DeviceSpecific<T>> per_device_op_state() {
  return {OpArgRefType::PER_DEVICE_OP_STATE};
}

OpArgRef<ParallelTensorShape> input_parallel_tensor_shape(int idx) {
  return {OpArgRefType::PARALLEL_TENSOR_SHAPE};
}

} // namespace FlexFlow

#endif
