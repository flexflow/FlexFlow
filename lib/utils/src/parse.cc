#include "utils/parse.h"
#include "utils/containers.h"
#include "utils/exception.h"
#include "utils/variant.h"
#include <vector>

namespace FlexFlow {

// currently we only support "--xx" or "-x"
std::string parseKey(std::string const &arg) {
  if (arg.substr(0, 2) == "--") {
    return arg.substr(2);
  } else if (arg.substr(0, 1) == "-") {
    return arg;
  }
  throw mk_runtime_error("parse invalid args: " + arg);
}

ArgsParser parse_args(ArgsParser const &mArgs, int argc, char const **argv) {
  int i = 1;
  ArgsParser result;
  std::vector<std::string> required_args_passed;
  for (auto const &[key, arg] : mArgs.mArguments) {
    result.mArguments[key] = arg;
  }
  result.num_required_args = mArgs.num_required_args;

  while (i < argc) {
    std::string key = parseKey(argv[i]);
    if (key == "help" || key == "h") {
      exit(1);
    }

    if (mArgs.mArguments.count(key) && mArgs.mArguments.at(key).is_store_true) {
      result.mArguments[key].value = "true";
      result.mArguments[key].is_store_true = true;
      result.mArguments[key].is_store_passed = true;
      i++;
      continue;
    }

    if (i + 1 < argc && argv[i + 1][0] != '-') {
      if (result.mArguments.count(key)) {
        if (result.mArguments.at(key).is_optional) {
          result.mArguments[key].value = argv[i + 1];
        } else {
          // required args
          result.mArguments[key].value = argv[i + 1];
          result.pass_required_args++;
          required_args_passed.push_back(key);
        }
      } else {
        throw mk_runtime_error("invalid args: " + key + " does not exist");
      }
      i += 2;
    } else {
      if (result.mArguments.count(key) &&
          !result.mArguments.at(key).is_store_true) {
        throw mk_runtime_error("required args: " + key + " needs a value");
      }
      i++;
    }
  }
  if (result.num_required_args != result.pass_required_args) {
    std::vector<std::string> missing_args;
    for (auto const &[key, arg] : mArgs.mArguments) {
      if (!arg.is_optional) { // required args
        if (std::find(required_args_passed.begin(),
                      required_args_passed.end(),
                      key) == required_args_passed.end()) {
          missing_args.push_back(key);
        }
      }
    }
    // std::string missing_args_str = "";
    for (auto const &arg : missing_args) {
      // missing_args_str +=  arg + "  " ;
      std::cout << "missing_args:" << arg << std::endl;
    }
    throw mk_runtime_error("some required args are not passed");
  }

  return result;
}

// default_value is std::nullopt
template <typename T>
CmdlineArgRef<T> add_required_argument(ArgsParser &parser,
                                       std::string const &key,
                                       std::optional<T> const &default_value,
                                       std::string const &description,
                                       bool is_store_true = false) {
  std::string parse_key = parseKey(key);
  parser.mArguments[parse_key].description = description;
  parser.mArguments[parse_key].is_store_true = is_store_true;
  parser.num_required_args++;
  parser.mArguments[parse_key].is_optional = false;
  return CmdlineArgRef<T>{parse_key, T{}};
}

template <typename T>
CmdlineArgRef<T> add_optional_argument(ArgsParser &parser,
                                       std::string const &key,
                                       std::optional<T> const &default_value,
                                       std::string const &description,
                                       bool is_store_true = false) {
  std::string parse_key = parseKey(key);
  parser.mArguments[parse_key].description = description;
  if (default_value
          .has_value()) { // Use has_value() to check if there's a value
    parser.mArguments[parse_key].value =
        std::to_string(default_value.value()); // Convert the value to string
    parser.mArguments[parse_key].default_value = true;
    parser.mArguments[parse_key].is_store_true = is_store_true;
    parser.mArguments[parse_key].is_optional = true;
    return CmdlineArgRef<T>{parse_key, default_value.value()};
  }
  return CmdlineArgRef<T>{parse_key, T{}};
}

T get(ArgsParser const &parser, CmdlineArgRef<T> const &ref) {
  std::string key = ref.key;
  if (parser.mArguments.count(key)) {
    if (parser.mArguments.at(key).is_store_true) {
      if (parser.mArguments.at(key).is_store_passed) {
        return true;
      } else {
        return false;
      }
    } else {
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
