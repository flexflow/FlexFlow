#include "local-execution/init_task_impl_function.h"

namespace FlexFlow {

bool InitTaskImplFunction::operator==(InitTaskImplFunction const &other) const {
  return this->function_ptr == other.function_ptr;
}

bool InitTaskImplFunction::operator!=(InitTaskImplFunction const &other) const {
  return this->function_ptr != other.function_ptr;
}

bool InitTaskImplFunction::operator<(InitTaskImplFunction const &other) const {
  return this->function_ptr < other.function_ptr;
}

bool InitTaskImplFunction::operator>(InitTaskImplFunction const &other) const {
  return this->function_ptr > other.function_ptr;
}

bool InitTaskImplFunction::operator<=(InitTaskImplFunction const &other) const {
  return this->function_ptr <= other.function_ptr;
}

bool InitTaskImplFunction::operator>=(InitTaskImplFunction const &other) const {
  return this->function_ptr >= other.function_ptr;
}

std::string format_as(InitTaskImplFunction const &x) {
  std::ostringstream oss;
  oss << "<InitTaskImplFunction";
  oss << " function_ptr=" << reinterpret_cast<void *>(x.function_ptr);
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, InitTaskImplFunction const &x) {
  return s << fmt::to_string(x);
}

} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::InitTaskImplFunction>::operator()(
    ::FlexFlow::InitTaskImplFunction const &x) const {
  return std::hash<decltype(x.function_ptr)>{}(x.function_ptr);
}
} // namespace std
