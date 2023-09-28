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

template <typename T>
struct CmdlineArgRef {
  std::string key;
  T value;
};

class ArgsParser {
private:
  std::unordered_map<std::string, std::string> mArgs;
  std::unordered_map<std::string, AllowedArgTypes> mDefaultValues;
  std::unordered_map<std::string, std::string> mDescriptions;

public:
  ArgsParser() = default;
  void parse_args(int argc, char **argv);

  template <typename T>
  CmdlineArgRef<T> add_argument(std::string const &key,
                         T const &value,
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
