#include "utils/fmt.h"
#include "utils/containers.h"
#include "utils/string.h"

namespace FlexFlow {

std::string element_to_string(char const s[]) {
  return surrounded('"', quoted(s, '\\', '"'));
}

template <>
std::string element_to_string(std::string const &s) {
  return surrounded('"', quoted(s, '\\', '"'));
}

template <>
std::string element_to_string(char const &c) {
  return surrounded('\'', quoted(std::string{c}, '\\', '\''));
}

} // namespace FlexFlow
