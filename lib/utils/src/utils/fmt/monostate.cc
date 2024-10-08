#include "utils/fmt/monostate.h"

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s, std::monostate const &m) {
  return (s << fmt::to_string(m));
}

} // namespace FlexFlow
