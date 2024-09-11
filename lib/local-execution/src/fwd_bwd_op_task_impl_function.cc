#include "local-execution/fwd_bwd_op_task_impl_function.h"

namespace FlexFlow {

bool FwdBwdOpTaskImplFunction::operator==(
    FwdBwdOpTaskImplFunction const &other) const {
  return this->function_ptr == other.function_ptr;
}

bool FwdBwdOpTaskImplFunction::operator!=(
    FwdBwdOpTaskImplFunction const &other) const {
  return this->function_ptr != other.function_ptr;
}

bool FwdBwdOpTaskImplFunction::operator<(
    FwdBwdOpTaskImplFunction const &other) const {
  return this->function_ptr < other.function_ptr;
}

bool FwdBwdOpTaskImplFunction::operator>(
    FwdBwdOpTaskImplFunction const &other) const {
  return this->function_ptr > other.function_ptr;
}

bool FwdBwdOpTaskImplFunction::operator<=(
    FwdBwdOpTaskImplFunction const &other) const {
  return this->function_ptr <= other.function_ptr;
}

bool FwdBwdOpTaskImplFunction::operator>=(
    FwdBwdOpTaskImplFunction const &other) const {
  return this->function_ptr >= other.function_ptr;
}

std::string format_as(FwdBwdOpTaskImplFunction const &x) {
  std::ostringstream oss;
  oss << "<FwdBwdOpTaskImplFunction";
  oss << " function_ptr=" << reinterpret_cast<void *>(x.function_ptr);
  oss << ">";
  return oss.str();
}

std::ostream &operator<<(std::ostream &s, FwdBwdOpTaskImplFunction const &x) {
  return s << fmt::to_string(x);
}

} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::FwdBwdOpTaskImplFunction>::operator()(
    ::FlexFlow::FwdBwdOpTaskImplFunction const &x) const {
  return std::hash<decltype(x.function_ptr)>{}(x.function_ptr);
}
} // namespace std
