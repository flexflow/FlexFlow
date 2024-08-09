#include "local-execution/init_task_impl_function.h"

namespace FlexFlow
{

bool InitTaskImplFunction::operator==(InitTaskImplFunction const & other) const {
  return this->init_task_impl_function == other.init_task_impl_function;
}

bool InitTaskImplFunction::operator!=(InitTaskImplFunction const & other) const {
  return this->init_task_impl_function != other.init_task_impl_function;
}
  
std::string format_as(InitTaskImplFunction const &x) {
  std::ostringstream oss;
  oss << "<InitTaskImplFunction";
  oss << " init_task_impl_function=" << x.init_task_impl_function;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, InitTaskImplFunction const &x) {
  return s << fmt::to_string(x);
}


} // namespace FlexFlow
