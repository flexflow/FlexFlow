#ifndef _FLEXFLOW_LIB_UTILS_STRING_EXTRA_INCLUDE_UTILS_STRING_EXTRA_SURROUNDED_H
#define _FLEXFLOW_LIB_UTILS_STRING_EXTRA_INCLUDE_UTILS_STRING_EXTRA_SURROUNDED_H

#include <string>

namespace FlexFlow {

std::string surrounded(char pref_and_post, std::string const &s);
std::string surrounded(char prefix, char postfix, std::string const &s);
std::string surrounded(std::string const &pre_and_post, std::string const &s);
std::string surrounded(std::string const &prefix,
                       std::string const &postfix,
                       std::string const &s);

} // namespace FlexFlow

#endif
