#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_ARG_REF_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_ARG_REF_H

#include "local-execution/arg_ref.h"
#include "local-execution/device_specific.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

enum class OpArgRefType { PER_DEVICE_OP_STATE, PARALLEL_TENSOR_SHAPE };

template <typename T>
using OpArgRef = ArgRef<OpArgRefType, T>;

using OpArgRefSpec = ArgRefSpec<OpArgRefType>;

template <typename T>
OpArgRef<DeviceSpecific<T>> per_device_op_state();

OpArgRef<ParallelTensorShape> input_parallel_tensor_shape(int idx);

} // namespace FlexFlow

#endif
