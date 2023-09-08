#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_STRING_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_STRING_H

#include <string>
#include <unordered_set>

namespace FlexFlow {

std::string surrounded(char pref_and_post, std::string const &s);
std::string surrounded(char prefix, char postfix, std::string const &s);
std::string surrounded(std::string const &pre_and_post, std::string const &s);
std::string surrounded(std::string const &prefix,
                       std::string const &postfix,
                       std::string const &s);

std::string quoted(std::string const &s, char escape_char);
std::string quoted(std::string const &s, char escape_char, char to_escape);
std::string quoted(std::string const &s,
                   char escape_char,
                   std::unordered_set<char> const &to_escape);

} // namespace FlexFlow

#endif
