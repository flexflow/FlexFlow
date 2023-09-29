#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H

#include "utils/containers.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include "utils/variant.h"
#include <ostream>
#include <string>
#include <unordered_map>

namespace FlexFlow {

using AllowedArgTypes =
    variant<int, bool, float, std::string>; // we can add more types here

template <typename T>
struct CmdlineArgRef {
  std::string key;
  T value;
};

struct Argument {
  optional<std::string> value; // Change value type to optional<string>
  std::string description;
  bool default_value;
};

struct ArgsParser {
  std::unordered_map<std::string, Argument> mArguments;
};

// currently we only support "--xx" or "-x"
std::string parseKey(std::string arg);

void parse_args(ArgsParser &mArgs, int argc, char **argv);

template <typename T>
CmdlineArgRef<T> add_argument(ArgsParser &parser,
                              std::string key,
                              optional<T> default_value,
                              std::string const &description);

template <typename T>
T get(ArgsParser const &parser, CmdlineArgRef<T> const &ref);

void showDescriptions(ArgsParser const &parser);

template <typename T>
T convert(std::string const &s);

template <>
int convert<int>(std::string const &s);

template <>
float convert<float>(std::string const &s);

template <>
bool convert<bool>(std::string const &s);

} // namespace FlexFlow

#endif
