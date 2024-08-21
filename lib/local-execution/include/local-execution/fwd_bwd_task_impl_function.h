#ifndef _FLEXFLOW_LOCAL_EXECUTION_FWD_BWD_TASK_IMPL_FUNCTION_H
#define _FLEXFLOW_LOCAL_EXECUTION_FWD_BWD_TASK_IMPL_FUNCTION_H

#include "local-execution/task_argument_accessor.h"

namespace FlexFlow {

struct FwdBwdTaskImplFunction {

  std::optional<float> (*function_ptr)(TaskArgumentAccessor const &);

  bool operator==(FwdBwdTaskImplFunction const &) const;
  bool operator!=(FwdBwdTaskImplFunction const &) const;
};

std::string format_as(FwdBwdTaskImplFunction const &x);
std::ostream &operator<<(std::ostream &s, FwdBwdTaskImplFunction const &x);

} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::FwdBwdTaskImplFunction> {
  size_t operator()(::FlexFlow::FwdBwdTaskImplFunction const &) const;
};
} // namespace std

#endif
