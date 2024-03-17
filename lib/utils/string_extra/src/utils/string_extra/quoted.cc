#include "utils/string_extra/quoted.h"
#include "utils/algorithms/flatmap.h"
#include "utils/algorithms/generic/contains.h"
#include <string>
#include <unordered_set>

namespace FlexFlow {

std::string quoted(std::string const &s, char escape_char) {
  return quoted(s, escape_char, std::unordered_set<char>{});
}

std::string quoted(std::string const &s, char escape_char, char to_escape) {
  return quoted(s, escape_char, std::unordered_set{to_escape});
}

std::string quoted(std::string const &s,
                   char escape_char,
                   std::unordered_set<char> const &to_escape) {
  return flatmap(s, [&](char c) -> std::string {
    if (c == escape_char || contains(to_escape, c)) {
      return {escape_char, c};
    } else {
      return {c};
    }
  });
}

} // namespace FlexFlow
