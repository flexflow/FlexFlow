#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H

#include "utils/exception.h"
#include "utils/variant.h"
#include <ostream>
#include <string>
#include <unordered_map>
namespace FlexFlow {

using AllowedArgTypes = variant<int, bool, float, std::string>;

std::string parseKey(std::string const &arg) const {
  if (arg.substr(0, 2) == "--") {
    return arg.substr(2);
  } else {
    return arg;
  }
}

class ArgsParser {
private:
  std::unordered_map<std::string, std::string> mArgs;
  std::unordered_map<std::string, AllowedArgTypes> mDefaultValues;
  std::unordered_map<std::string, std::string> mDescriptions;

public:
  ArgsParser() = default;
  void parse_args(int argc, char **argv);

  template <typename T>
  class ArgumentReference {
  public:
    ArgumentReference(AllowedArgTypes const &defaultValue,
                      std::string const &description)
        : defaultValue(defaultValue), description(description), key(key) {}

    AllowedArgTypes const &default_value() const {
      return default_value;
    }

  private:
    AllowedArgTypes defaultValue;
    std::string description;
    std::string key;
  };

  template <typename T>
  T get_from_variant(AllowedArgTypes const &v) const;

  template <typename T>
  ArgumentReference<T> add_argument(std::string const &key,
                                    AllowedArgTypes const &value,
                                    std::string const &description);

  template <typename T>
  T get(ArgumentReference<T> const &arg_ref) const;

  void showDescriptions() const;

  template <typename T>
  T convert(std::string const &s) const;

  friend std::ostream &operator<<(std::ostream &out, ArgsParser const &args);
};

} // namespace FlexFlow

#endif
