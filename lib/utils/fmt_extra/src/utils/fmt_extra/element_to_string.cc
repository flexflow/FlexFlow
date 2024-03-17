#include "utils/fmt_extra/element_to_string.h"
#include "utils/string_extra/quoted.h"
#include "utils/string_extra/surrounded.h"

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

}
