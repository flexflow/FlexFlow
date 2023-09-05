#include "utils/parse.h"
#include "utils/containers.h"
#include "utils/variant.h"

namespace FlexFlow{

std::string parseKey(std::string arg) {
    if (arg.substr(0, 2) == "--") {
      return arg.substr(2);
    } else {
      return arg;
    }
  }

void ArgsParser::add_argument(std::string const &key,
                    AllowedArgTypes const &value,
                    std::string const &description) {
    mDefaultValues[parseKey(key)] = value;
    mDescriptions[key] = description;
  }

template <>
int ArgsParser::convert<int>(std::string const &s) const {
  return std::stoi(s);
}

template <>
float ArgsParser::convert<float>(std::string const &s) const {
  return std::stof(s);
}

template <>
bool ArgsParser::convert<bool>(std::string const &s) const {
  return s == "true" || s == "1" || s == "yes";
}

void ArgsParser::showDescriptions() const {
    for (auto const &kv : mDescriptions) {
      std::cerr  << kv.first << ": " << kv.second << std::endl;
    }
  }

//TODO(lambda):in the future, we will use fmt to format the output
std::ostream &operator<<(std::ostream &out, ArgsParser const &args) {
  args.showDescriptions();
  return out;
}

} // namespace FlexFlow
