#ifndef _FLEXFLOW_LOCAL_EXECUTION_ARG_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_ARG_BACKING_H

#include "kernels/linear_kernels.h"
#include "parallel_tensor_shape.h"
#include "slot_id.h"
#include "device_specific.h"
#include <unordered_map>
#include <variant>

namespace FlexFlow {

// TODO: define device state variant in another file
using DeviceStates = std::variant<LinearPerDeviceState>;

template <typename DeviceState>
struct OpArgBacking {

  std::unordered_map<slot_id, ParallelTensorShape> tensor_shapes;
  std::pair<slot_id, optional<DeviceSpecific<DeviceState>>> per_device_op_state;
  std::pair<slot_id, ProfilingSettings> profiling_settings;
  std::pair<slot_id, DeviceSpecific<PerDeviceFFHandle>> profiling_settings;
  std::pair<slot_id, FFIterationConfig> profiling_settings;

};

using ArgBackingMapping = std::unordered_map<operator_guid_t, OpArgBacking<DeviceStates>> per_op_arg_backing;

} // namespace FlexFlow

#endif