#include "local-execution/fwd_bwd_task_impl_function.h"

namespace FlexFlow
{

bool FwdBwdTaskImplFunction::operator==(FwdBwdTaskImplFunction const & other) const {
  return this->fwd_bwd_task_impl_function == other.fwd_bwd_task_impl_function;
}

bool FwdBwdTaskImplFunction::operator!=(FwdBwdTaskImplFunction const & other) const {
  return this->fwd_bwd_task_impl_function != other.fwd_bwd_task_impl_function;
}
  
std::string format_as(FwdBwdTaskImplFunction const &x) {
  std::ostringstream oss;
  oss << "<FwdBwdTaskImplFunction";
  oss << " fwd_bwd_task_impl_function=" << x.fwd_bwd_task_impl_function;
  oss << ">";
  return oss.str();
}

std::ostream &operator<<(std::ostream &s, FwdBwdTaskImplFunction const &x) {
  return s << fmt::to_string(x);
}


} // namespace FlexFlow
