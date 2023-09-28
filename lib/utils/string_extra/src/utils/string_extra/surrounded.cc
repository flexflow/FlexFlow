#include "utils/string_extra/surrounded.h"
#include <sstream>

namespace FlexFlow {

std::string surrounded(char pre_and_post, std::string const &s) {
  return surrounded(pre_and_post, pre_and_post, s);
}

std::string surrounded(char prefix, char postfix, std::string const &s) {
  return surrounded(std::string{prefix}, std::string{postfix}, s);
}

std::string surrounded(std::string const &pre_and_post, std::string const &s) {
  return surrounded(pre_and_post, pre_and_post, s);
}

std::string surrounded(std::string const &prefix,
                       std::string const &postfix,
                       std::string const &s) {
  std::ostringstream oss;
  oss << prefix << s << postfix;
  return oss.str();
}


}
