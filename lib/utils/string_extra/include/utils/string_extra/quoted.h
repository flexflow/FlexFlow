#ifndef _FLEXFLOW_LIB_UTILS_STRING_EXTRA_INCLUDE_UTILS_STRING_EXTRA_QUOTED_H
#define _FLEXFLOW_LIB_UTILS_STRING_EXTRA_INCLUDE_UTILS_STRING_EXTRA_QUOTED_H

#include <string>
#include <unordered_set>

namespace FlexFlow {

std::string quoted(std::string const &s, char escape_char);
std::string quoted(std::string const &s, char escape_char, char to_escape);
std::string quoted(std::string const &s,
                   char escape_char,
                   std::unordered_set<char> const &to_escape);

} // namespace FlexFlow

#endif
