#ifndef _FLEXFLOW_LOCAL_EXECUTION_INIT_TASK_IMPL_FUNCTION_H
#define _FLEXFLOW_LOCAL_EXECUTION_INIT_TASK_IMPL_FUNCTION_H

#include "local-execution/device_specific_device_states.dtg.h"
#include "local-execution/task_argument_accessor.h"

namespace FlexFlow {

struct InitTaskImplFunction {

  DeviceSpecificDeviceStates (*function_ptr)(TaskArgumentAccessor const &);

  bool operator==(InitTaskImplFunction const &) const;
  bool operator!=(InitTaskImplFunction const &) const;
  bool operator<(InitTaskImplFunction const &) const;
  bool operator>(InitTaskImplFunction const &) const;
  bool operator<=(InitTaskImplFunction const &) const;
  bool operator>=(InitTaskImplFunction const &) const;
};

std::string format_as(InitTaskImplFunction const &x);
std::ostream &operator<<(std::ostream &s, InitTaskImplFunction const &x);

} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::InitTaskImplFunction> {
  size_t operator()(::FlexFlow::InitTaskImplFunction const &) const;
};
} // namespace std

#endif
