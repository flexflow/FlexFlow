#include "utils/fmt/optional.h"

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s, std::nullopt_t) {
  return (s << std::string{"nullopt"});
}

} // namespace FlexFlow
