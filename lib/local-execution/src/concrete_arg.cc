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

} // namespace FlexFlow
