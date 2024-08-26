#include "local-execution/concrete_arg.h"

namespace FlexFlow {

bool ConcreteArgSpec::operator==(ConcreteArgSpec const &other) const {
  return this->tie() == other.tie();
}

bool ConcreteArgSpec::operator!=(ConcreteArgSpec const &other) const {
  return this->tie() != other.tie();
}

std::tuple<std::type_index const &> ConcreteArgSpec::tie() const {
  return std::tie(this->type_idx);
}

std::string format_as(ConcreteArgSpec const &x) {
  std::ostringstream oss;
  oss << "<ConcreteArgSpec";
  oss << " type_index=" << x.get_type_index().name();
  oss << ">";
  return oss.str();
}

std::ostream &operator<<(std::ostream &s, ConcreteArgSpec const &x) {
  return s << fmt::to_string(x);
}

} // namespace FlexFlow
