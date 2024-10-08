#include "local-execution/generic_task_impl_function.h"

namespace FlexFlow {

bool GenericTaskImplFunction::operator==(
    GenericTaskImplFunction const &other) const {
  return this->function_ptr == other.function_ptr;
}

bool GenericTaskImplFunction::operator!=(
    GenericTaskImplFunction const &other) const {
  return this->function_ptr != other.function_ptr;
}

bool GenericTaskImplFunction::operator<(
    GenericTaskImplFunction const &other) const {
  return this->function_ptr < other.function_ptr;
}

bool GenericTaskImplFunction::operator>(
    GenericTaskImplFunction const &other) const {
  return this->function_ptr > other.function_ptr;
}

bool GenericTaskImplFunction::operator<=(
    GenericTaskImplFunction const &other) const {
  return this->function_ptr <= other.function_ptr;
}

bool GenericTaskImplFunction::operator>=(
    GenericTaskImplFunction const &other) const {
  return this->function_ptr >= other.function_ptr;
}

std::string format_as(GenericTaskImplFunction const &x) {
  std::ostringstream oss;
  oss << "<GenericTaskImplFunction";
  oss << " function_ptr=" << reinterpret_cast<void *>(x.function_ptr);
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, GenericTaskImplFunction const &x) {
  return s << fmt::to_string(x);
}

} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::GenericTaskImplFunction>::operator()(
    ::FlexFlow::GenericTaskImplFunction const &x) const {
  return std::hash<decltype(x.function_ptr)>{}(x.function_ptr);
}
} // namespace std
