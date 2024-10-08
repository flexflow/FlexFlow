#ifndef _FLEXFLOW_LOCAL_EXECUTION_FWD_BWD_TASK_IMPL_FUNCTION_H
#define _FLEXFLOW_LOCAL_EXECUTION_FWD_BWD_TASK_IMPL_FUNCTION_H

#include "local-execution/task_argument_accessor.h"

namespace FlexFlow {

struct FwdBwdOpTaskImplFunction {

  std::optional<float> (*function_ptr)(TaskArgumentAccessor const &);

  bool operator==(FwdBwdOpTaskImplFunction const &) const;
  bool operator!=(FwdBwdOpTaskImplFunction const &) const;
  bool operator<(FwdBwdOpTaskImplFunction const &) const;
  bool operator>(FwdBwdOpTaskImplFunction const &) const;
  bool operator<=(FwdBwdOpTaskImplFunction const &) const;
  bool operator>=(FwdBwdOpTaskImplFunction const &) const;
};

std::string format_as(FwdBwdOpTaskImplFunction const &x);
std::ostream &operator<<(std::ostream &s, FwdBwdOpTaskImplFunction const &x);

} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::FwdBwdOpTaskImplFunction> {
  size_t operator()(::FlexFlow::FwdBwdOpTaskImplFunction const &) const;
};
} // namespace std

#endif
