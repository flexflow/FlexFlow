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
  std::vector<std::string> optional_args_passed;
  for (auto const &[key, arg] : mArgs.requeiredArguments) {
    result.requeiredArguments[key] = arg;
  }
  for (auto const &[key, arg] : mArgs.optionalArguments) {
    result.optionalArguments[key] = arg;
  }
  result.num_optional_args = mArgs.num_optional_args;
  while (i < argc) {
    std::string key = parseKey(argv[i]);
    if (key == "help" || key == "h") {
      exit(1);
    }

    if (mArgs.requeiredArguments.count(key) &&
        mArgs.requeiredArguments.at(key).is_store_true) {
      result.requeiredArguments[key].value = "true";
      result.requeiredArguments[key].is_store_true = true;
      result.requeiredArguments[key].is_store_passed = true;
      i++;
      continue;
    }

    if (i + 1 < argc && argv[i + 1][0] != '-') {
      if (result.requeiredArguments.count(key)) {
        result.requeiredArguments[key].value = argv[i + 1];
      } else if (result.optionalArguments.count(key)) {
        result.optionalArguments[key].value = argv[i + 1];
        result.pass_optional_args++;
        optional_args_passed.push_back(key);
      } else {
        throw mk_runtime_error("invalid args: " + key + " does not exist");
      }
      i += 2;
    } else {
      if (result.requeiredArguments.count(key) &&
          !result.requeiredArguments.at(key).is_store_true) {
        throw mk_runtime_error("required args: " + key + " needs a value");
      }
      i++;
    }
  }

  if (result.pass_optional_args != result.num_optional_args) {
    std::vector<std::string> missing_args;
    for (auto const &[key, arg] : mArgs.optionalArguments) {
      if (std::find(optional_args_passed.begin(),
                    optional_args_passed.end(),
                    key) == optional_args_passed.end()) {
        missing_args.push_back(key);
      }
    }
    std::string missing_args_str = "";
    for (auto const &arg : missing_args) {
      missing_args_str += arg + "  ";
    }
    throw mk_runtime_error("some optional args are not passed: " +
                           missing_args_str);
  }

  return result;
}

template <typename T>
CmdlineArgRef<T> add_required_argument(ArgsParser &parser,
                                       std::string const &key,
                                       std::optional<T> const &default_value,
                                       std::string const &description,
                                       bool is_store_true = false) {
  std::string parse_key = parseKey(key);
  parser.requeiredArguments[parse_key].description = description;
  if (default_value
          .has_value()) { // Use has_value() to check if there's a value
    parser.requeiredArguments[parse_key].value =
        std::to_string(default_value.value()); // Convert the value to string
    parser.requeiredArguments[parse_key].default_value = true;
    parser.requeiredArguments[parse_key].is_store_true = is_store_true;
    return CmdlineArgRef<T>{parse_key, default_value.value()};
  }
  return CmdlineArgRef<T>{parse_key, T{}};
}

// default_value is std::nullopt
template <typename T>
CmdlineArgRef<T> add_optional_argument(ArgsParser &parser,
                                       std::string const &key,
                                       std::optional<T> const &default_value,
                                       std::string const &description,
                                       bool is_store_true = false) {
  std::string parse_key = parseKey(key);
  parser.optionalArguments[parse_key].description = description;
  parser.optionalArguments[parse_key].is_store_true = is_store_true;
  parser.num_optional_args++;
  return CmdlineArgRef<T>{parse_key, T{}};
}

template <typename T>
T get(ArgsParser const &parser, CmdlineArgRef<T> const &ref) {
  std::string key = ref.key;
  if (parser.requeiredArguments.count(key)) {
    if (parser.requeiredArguments.at(key).is_store_true) {
      if (parser.requeiredArguments.at(key).is_store_passed) {
        return true;
      } else {
        return false;
      }
    } else if (parser.requeiredArguments.at(key).default_value ||
               parser.requeiredArguments.at(key).value.has_value()) {
      return convert<T>(parser.requeiredArguments.at(key).value.value());
    }
  } else if (parser.optionalArguments.count(key)) {
    if (parser.optionalArguments.at(key).default_value ||
        parser.optionalArguments.at(key).value.has_value()) {
      return convert<T>(parser.optionalArguments.at(key).value.value());
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
