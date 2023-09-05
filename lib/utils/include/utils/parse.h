#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H

#include "utils/containers.h"
#include "utils/exception.h"
#include "utils/variant.h"
#include <ostream>
#include <string>
#include <unordered_map>

namespace FlexFlow {

std::string parseKey(std::string arg);

using AllowedArgTypes = variant<int, bool, float, std::string>;

class ArgsParser {
private:
  std::unordered_map<std::string, std::string> mArgs;
  std::unordered_map<std::string, AllowedArgTypes> mDefaultValues;
  std::unordered_map<std::string, std::string> mDescriptions;

  std::string parseKey(std::string const &arg) const {
    if (arg.substr(0, 2) == "--") {
      return arg.substr(2);
    } else {
      return arg;
    }
  }

public:
  ArgsParser() = default;
  void parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; i += 2) {
      std::string key = parseKey(argv[i]);
      if (key == "help" || key == "h") {
        showDescriptions();
        exit(0);
      }
      mArgs[key] = argv[i + 1];
    }
  }

  template <typename T>
  T get_from_variant(AllowedArgTypes const &v) const;

  void add_argument(std::string const &key,
                    AllowedArgTypes const &value,
                    std::string const &description);

  template <typename T>
  T get(std::string const &key) const {
    if (contains_key(mArgs, key)) {
      return convert<T>(mArgs.at(key));
    } else {
      if (contains_key(mDefaultValues, key)) {
        return mpark::get<T>(mDefaultValues.at(key));
      }
    }
    throw mk_runtime_error("Key not found: " + key);
  }

  void showDescriptions() const;

  template <typename T>
  T convert(std::string const &s) const;

  friend std::ostream &operator<<(std::ostream &out, ArgsParser const &args);
};

} // namespace FlexFlow

#endif
