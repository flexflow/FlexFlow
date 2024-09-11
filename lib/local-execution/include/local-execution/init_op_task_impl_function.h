#ifndef _FLEXFLOW_LOCAL_EXECUTION_INIT_TASK_IMPL_FUNCTION_H
#define _FLEXFLOW_LOCAL_EXECUTION_INIT_TASK_IMPL_FUNCTION_H

#include "local-execution/device_specific_device_states.dtg.h"
#include "local-execution/task_argument_accessor.h"

namespace FlexFlow {

struct InitOpTaskImplFunction {

  DeviceSpecificDeviceStates (*function_ptr)(TaskArgumentAccessor const &);

  bool operator==(InitOpTaskImplFunction const &) const;
  bool operator!=(InitOpTaskImplFunction const &) const;
  bool operator<(InitOpTaskImplFunction const &) const;
  bool operator>(InitOpTaskImplFunction const &) const;
  bool operator<=(InitOpTaskImplFunction const &) const;
  bool operator>=(InitOpTaskImplFunction const &) const;
};

std::string format_as(InitOpTaskImplFunction const &x);
std::ostream &operator<<(std::ostream &s, InitOpTaskImplFunction const &x);

} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::InitOpTaskImplFunction> {
  size_t operator()(::FlexFlow::InitOpTaskImplFunction const &) const;
};
} // namespace std

#endif
