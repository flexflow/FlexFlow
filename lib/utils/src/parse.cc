#include "utils/parse.h"
#include "utils/containers.h"
#include "utils/exception.h"
#include "utils/variant.h"
namespace FlexFlow {

// currently we only support "--xx" or "-x"
std::string parseKey(std::string arg) {
  if (arg.substr(0, 2) == "--") {
    return arg.substr(2);
  } else if (arg.substr(0, 1) == "-") {
    return arg;
  }
  throw mk_runtime_error("parse invalid args: " + arg);
}

void parse_args(ArgsParser &mArgs, int argc, char **argv) {
  for (int i = 1; i < argc; i += 2) {
    std::string key = parseKey(argv[i]);
    if (key == "help" || key == "h") {
      showDescriptions(mArgs);
      exit(1);
    }
    mArgs.mArguments[key].value = argv[i + 1];
  }
}

template <typename T>
CmdlineArgRef<T> add_argument(ArgsParser &parser,
                              std::string key,
                              std::optional<T> default_value,
                              std::string const &description) {
  key = parseKey(key);
  parser.mArguments[key].description = description;
  if (default_value
          .has_value()) { // Use has_value() to check if there's a value
    parser.mArguments[key].value =
        std::to_string(default_value.value()); // Convert the value to string
    parser.mArguments[key].default_value = true;
    return CmdlineArgRef<T>{key, default_value.value()};
  }
  return CmdlineArgRef<T>{key, T{}};
}

template <typename T>
T get(ArgsParser const &parser, CmdlineArgRef<T> const &ref) {
  std::string key = ref.key;
  if (parser.mArguments.count(key)) {
    if (parser.mArguments.at(key).default_value ||
        parser.mArguments.at(key).value.has_value()) {
      return convert<T>(parser.mArguments.at(key).value.value());
    }
  }
  throw mk_runtime_error("invalid args: " + ref.key);
}

void showDescriptions(ArgsParser const &parser) {
  NOT_IMPLEMENTED(); // TODO(lambda) I will use fmt to implement
}

template <>
int convert<int>(std::string const &s) {
  return std::stoi(s);
}

template <>
float convert<float>(std::string const &s) {
  return std::stof(s);
}

template <>
bool convert<bool>(std::string const &s) {
  return s == "true" || s == "1" || s == "yes";
}

} // namespace FlexFlow
