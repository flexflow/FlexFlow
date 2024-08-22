#include "local-execution/fwd_bwd_task_impl_function.h"

namespace FlexFlow {

bool FwdBwdTaskImplFunction::operator==(
    FwdBwdTaskImplFunction const &other) const {
  return this->function_ptr == other.function_ptr;
}

bool FwdBwdTaskImplFunction::operator!=(
    FwdBwdTaskImplFunction const &other) const {
  return this->function_ptr != other.function_ptr;
}

bool FwdBwdTaskImplFunction::operator<(
    FwdBwdTaskImplFunction const &other) const {
  return this->function_ptr < other.function_ptr;
}

bool FwdBwdTaskImplFunction::operator>(
    FwdBwdTaskImplFunction const &other) const {
  return this->function_ptr > other.function_ptr;
}

bool FwdBwdTaskImplFunction::operator<=(
    FwdBwdTaskImplFunction const &other) const {
  return this->function_ptr <= other.function_ptr;
}

bool FwdBwdTaskImplFunction::operator>=(
    FwdBwdTaskImplFunction const &other) const {
  return this->function_ptr >= other.function_ptr;
}

std::string format_as(FwdBwdTaskImplFunction const &x) {
  std::ostringstream oss;
  oss << "<FwdBwdTaskImplFunction";
  oss << " function_ptr=" << reinterpret_cast<void *>(x.function_ptr);
  oss << ">";
  return oss.str();
}

std::ostream &operator<<(std::ostream &s, FwdBwdTaskImplFunction const &x) {
  return s << fmt::to_string(x);
}

} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::FwdBwdTaskImplFunction>::operator()(
    ::FlexFlow::FwdBwdTaskImplFunction const &x) const {
  return std::hash<decltype(x.function_ptr)>{}(x.function_ptr);
}
} // namespace std
