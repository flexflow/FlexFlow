#include "local-execution/op_arg_ref.h"

namespace FlexFlow {

template <typename T>
OpArgRef<DeviceSpecific<T>> per_device_op_state() {
  return {OpArgRefType::PER_DEVICE_OP_STATE};
}

OpArgRef<ParallelTensorShape> input_parallel_tensor_shape(int idx) {
  return {OpArgRefType::PARALLEL_TENSOR_SHAPE};
}

} // namespace FlexFlow
