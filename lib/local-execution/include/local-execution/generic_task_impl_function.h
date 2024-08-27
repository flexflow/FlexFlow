#ifndef _FLEXFLOW_LOCAL_EXECUTION_GENERIC_TASK_IMPL_FUNCTION_H
#define _FLEXFLOW_LOCAL_EXECUTION_GENERIC_TASK_IMPL_FUNCTION_H

#include "local-execution/device_specific_device_states.dtg.h"
#include "local-execution/task_argument_accessor.h"

namespace FlexFlow {

struct GenericTaskImplFunction {

  void (*function_ptr)(TaskArgumentAccessor const &);

  bool operator==(GenericTaskImplFunction const &) const;
  bool operator!=(GenericTaskImplFunction const &) const;
  bool operator<(GenericTaskImplFunction const &) const;
  bool operator>(GenericTaskImplFunction const &) const;
  bool operator<=(GenericTaskImplFunction const &) const;
  bool operator>=(GenericTaskImplFunction const &) const;
};

std::string format_as(GenericTaskImplFunction const &x);
std::ostream &operator<<(std::ostream &s, GenericTaskImplFunction const &x);

} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::GenericTaskImplFunction> {
  size_t operator()(::FlexFlow::GenericTaskImplFunction const &) const;
};
} // namespace std

#endif
