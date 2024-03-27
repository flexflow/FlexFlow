#ifndef _FLEXFLOW_LOCAL_EXECUTION_ARG_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_ARG_BACKING_H

#include "kernels/linear_kernels.h"
#include "kernels/profiling.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/operator_guid_t.h"
#include "config.h"
#include "slot_id.h"
#include "device_specific.h"
#include <unordered_map>
#include <variant>

namespace FlexFlow {

// TODO: define device state variant in another file
using DeviceStates = std::variant<LinearPerDeviceState>;

struct OpArgBacking {

  std::unordered_map<slot_id, ParallelTensorShape> tensor_shapes;
  std::pair<slot_id, std::optional<DeviceSpecific<DeviceStates>>> per_device_op_state = std::nullopt;
  std::pair<slot_id, ProfilingSettings> profiling_settings;
  std::pair<slot_id, DeviceSpecific<PerDeviceFFHandle>> per_device_ff_handle;
  std::pair<slot_id, FFIterationConfig> iteration_config;

};

using ArgBackingMapping = std::unordered_map<operator_guid_t, OpArgBacking>;

} // namespace FlexFlow

#endif