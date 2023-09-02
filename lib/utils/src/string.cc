#include "utils/string.h"
#include "utils/containers.h"

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

std::string surrounded(std::string const &prefix, std::string const &postfix, std::string const &s) {
  std::ostringstream oss;
  oss << prefix << s << postfix;
  return oss.str();
}

std::string quoted(std::string const &s, char escape_char) {
  return quoted(s, escape_char, std::unordered_set<char>{});
}

std::string quoted(std::string const &s, char escape_char, char to_escape) {
  return quoted(s, escape_char, std::unordered_set{to_escape});
}

std::string quoted(std::string const &s, char escape_char, std::unordered_set<char> const &to_escape) {
  return flatmap(s, [&](char c) -> std::string {
    if (c == escape_char || contains(to_escape, c)) {
      return {escape_char, c};
    } else {
      return {c};
    }
  });
}

}
