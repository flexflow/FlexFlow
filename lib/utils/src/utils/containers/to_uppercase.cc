#include "utils/containers/to_uppercase.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::string to_uppercase(std::string const &s) {
  return transform(s, [](char c) -> char { return std::toupper(c); });
}

} // namespace FlexFlow
