#ifndef _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_ELEMENT_TO_STRING_H
#define _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_ELEMENT_TO_STRING_H

#include <fmt/format.h>
#include <string>

namespace FlexFlow {

template <typename T>
std::string element_to_string(T const &t) {
  return fmt::to_string(t);
}

std::string element_to_string(char const s[]);
template <>
std::string element_to_string(std::string const &s);
template <>
std::string element_to_string(char const &c);

} // namespace FlexFlow

#endif
