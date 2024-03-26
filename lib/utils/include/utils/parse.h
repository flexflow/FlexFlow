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
  std::optional<std::string> value; // Change value type to optional<string>
  std::string description;
  bool default_value = false;
  bool is_store_true =
      false; // Add a new field to indicate whether the argument is store_true
  bool is_store_passed =
      false; // Add a new field to indicate whether the argument is passed
  bool is_optional = false;
};

struct ArgsParser {
  std::unordered_map<std::string, Argument> mArguments;
  int num_required_args = 0;
  int pass_required_args = 0;
};

// currently we only support "--xx" or "-x"
std::string parseKey(std::string const &arg);

ArgsParser parse_args(ArgsParser const &mArgs, int argc, char const **argv)

    // default_value is std::nullopt for optional arguments
    template <typename T>
    CmdlineArgRef<T> add_required_argument(
        ArgsParser &parser,
        std::string const &key,
        std::optional<T> const &default_value,
        std::string const &description,
        bool is_store_true = false);

template <typename T>
CmdlineArgRef<T> add_optional_argument(ArgsParser &parser,
                                       std::string const &key,
                                       std::optional<T> const &default_value,
                                       std::string const &description,
                                       bool is_store_true = false);

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
