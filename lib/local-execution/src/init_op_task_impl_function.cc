#include "local-execution/init_op_task_impl_function.h"

namespace FlexFlow {

bool InitOpTaskImplFunction::operator==(
    InitOpTaskImplFunction const &other) const {
  return this->function_ptr == other.function_ptr;
}

bool InitOpTaskImplFunction::operator!=(
    InitOpTaskImplFunction const &other) const {
  return this->function_ptr != other.function_ptr;
}

bool InitOpTaskImplFunction::operator<(
    InitOpTaskImplFunction const &other) const {
  return this->function_ptr < other.function_ptr;
}

bool InitOpTaskImplFunction::operator>(
    InitOpTaskImplFunction const &other) const {
  return this->function_ptr > other.function_ptr;
}

bool InitOpTaskImplFunction::operator<=(
    InitOpTaskImplFunction const &other) const {
  return this->function_ptr <= other.function_ptr;
}

bool InitOpTaskImplFunction::operator>=(
    InitOpTaskImplFunction const &other) const {
  return this->function_ptr >= other.function_ptr;
}

std::string format_as(InitOpTaskImplFunction const &x) {
  std::ostringstream oss;
  oss << "<InitOpTaskImplFunction";
  oss << " function_ptr=" << reinterpret_cast<void *>(x.function_ptr);
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, InitOpTaskImplFunction const &x) {
  return s << fmt::to_string(x);
}

} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::InitOpTaskImplFunction>::operator()(
    ::FlexFlow::InitOpTaskImplFunction const &x) const {
  return std::hash<decltype(x.function_ptr)>{}(x.function_ptr);
}
} // namespace std
