#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_OP_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_OP_ARG_REF_H

#include "arg_ref.h"
#include "device_specific_arg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "runtime/config.h"

namespace FlexFlow {

enum class OpArgRefType {
  PER_DEVICE_OP_STATE,
  PARALLEL_TENSOR_SHAPE,
  ITERATION_CONFIG
};

template <typename T>
using OpArgRef = ArgRef<OpArgRefType, T>;

using OpArgRefSpec = ArgRefSpec<OpArgRefType>;

template <typename T>
OpArgRef<DeviceSpecificArg<T>> per_device_op_state() {
  return {OpArgRefType::PER_DEVICE_OP_STATE};
}

OpArgRef<ParallelTensorShape> input_parallel_tensor_shape(int idx) {
  return {OpArgRefType::PARALLEL_TENSOR_SHAPE};
}

OpArgRef<FFIterationConfig> iteration_config() {
  return {OpArgRefType::ITERATION_CONFIG};
}

} // namespace FlexFlow

#endif
