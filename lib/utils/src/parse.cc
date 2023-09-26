#include "utils/parse.h"
#include "utils/containers.h"
#include "utils/variant.h"

namespace FlexFlow {

std::string parseKey(std::string arg) {
  if (arg.substr(0, 2) == "--") {
    return arg.substr(2);
  } else {
    return arg;
  }
}

void ArgsParser::parse_args(int argc, char **argv) {
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
ArgRef<T> ArgsParser::add_argument(std::string const &key,
                                   T const &value,
                                   std::string const &description) {
  mDefaultValues[parseKey(key)] = value;
  mDescriptions[key] = description;
  return ArgRef<T>{key, value};
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
    std::cerr << *this << std::endl;
  }
}

// TODO(lambda):in the future, we will use fmt to format the output
std::ostream &operator<<(std::ostream &out, ArgsParser const &args) {
  args.showDescriptions();
  return out;
}

} // namespace FlexFlow
