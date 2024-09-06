#include "utils/cli/cli_spec.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/count.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/set_union.h"
#include "utils/exception.h"
#include "utils/containers/enumerate.h"
#include "utils/containers/transform.h"
#include "utils/integer_conversions.h"
#include "utils/containers/transform.h"
#include "utils/containers/to_uppercase.h"
#include "utils/fmt/unordered_set.h"

namespace FlexFlow {

CLISpec empty_cli_spec() {
  return CLISpec{{}, {}};
}

std::vector<CLIFlagKey> get_flag_keys(CLISpec const &cli) {
  return transform(count(cli.flags.size()), [](int idx) { return CLIFlagKey{idx}; });
}

CLIArgumentKey cli_add_flag(CLISpec &cli, CLIFlagSpec const &flag_spec) {
  cli.flags.push_back(flag_spec);

  return CLIArgumentKey{CLIFlagKey{int_from_size_t(cli.flags.size()) - 1}};
}

CLIArgumentKey cli_add_positional_argument(CLISpec &cli, CLIPositionalArgumentSpec const &arg) {
  cli.positional_arguments.push_back(arg);
  return CLIArgumentKey{CLIPositionalArgumentKey{int_from_size_t(cli.positional_arguments.size()) - 1}};
}

tl::expected<CLIFlagKey, std::string> cli_parse_flag(CLISpec const &cli, std::string const &arg) {
  for (auto const &[idx, flag_spec] : enumerate(cli.flags)) {
    CLIFlagKey key = CLIFlagKey{idx};
    if (("--" + flag_spec.long_flag) == arg) {
      return key;
    }
    
    if (flag_spec.short_flag.has_value()) {
      if ((std::string{"-"} + flag_spec.short_flag.value()) == arg) {
        return key;
      }
    }
  }

  return tl::unexpected(fmt::format("Encountered unknown flag {arg}"));
}

tl::expected<CLIParseResult, std::string> cli_parse(CLISpec const &cli, std::vector<std::string> const &args) {
  CLIParseResult result = CLIParseResult{
    generate_map(get_flag_keys(cli), [](CLIFlagKey const &) { return false; }),
    {},
  };

  int consumed_positional_args = 0;
  auto parse_positional_arg = [&](std::string const &arg) -> std::optional<std::string> {
    if (consumed_positional_args >= cli.positional_arguments.size()) {
      return fmt::format("Too many positional arguments: expected {}", cli.positional_arguments.size());
    }

    CLIPositionalArgumentSpec arg_spec = cli.positional_arguments.at(consumed_positional_args);
    
    if (arg_spec.options.has_value() && !contains(arg_spec.options.value(), arg)) {
      return fmt::format("Invalid option for positional argument: {}", arg);
    }

    result.positional_arguments.insert({CLIPositionalArgumentKey{consumed_positional_args}, arg});
    consumed_positional_args++;
    
    return std::nullopt;
  };

  for (int i = 1; i < args.size(); i++) {
    std::string arg = args.at(i);

    if (arg.at(0) == '-') {
      tl::expected<CLIFlagKey, std::string> parsed_flag = cli_parse_flag(cli, arg);       

      if (parsed_flag.has_value()) {
        result.flags.at(parsed_flag.value()) = true;
      }
    } else {
      std::optional<std::string> maybe_err_msg = parse_positional_arg(arg);
      if (maybe_err_msg.has_value()) {
        return tl::unexpected(maybe_err_msg.value());
      }
    }
  }

  if (consumed_positional_args != cli.positional_arguments.size()) {
    return tl::unexpected(fmt::format("Not enough positional arguments: found {}, expected {}", consumed_positional_args, cli.positional_arguments.size()));
  }

  return result;
}

tl::expected<CLIParseResult, std::string> cli_parse(CLISpec const &cli, int argc, char const * const *argv) {
  std::vector<std::string> args = {argv, argv+argc};

  return cli_parse(cli, args);
}

std::string cli_get_help_message(std::string const &program_name, CLISpec const &cli) {
  std::ostringstream oss;

  oss << "usage: " << program_name;
  oss << " [-h]";
  for (CLIFlagSpec const &flag_spec : cli.flags) {
    oss << " [--" << flag_spec.long_flag << "]";
  }
  for (CLIPositionalArgumentSpec const &pos_arg_spec : cli.positional_arguments) {
    if (pos_arg_spec.options.has_value()) {
      oss << " " << pos_arg_spec.options.value();
    } else {
      oss << " <" << pos_arg_spec.name << ">";
    }
  }

  oss << std::endl;
  oss << std::endl;
  oss << "positional arguments:" << std::endl;
  if (!cli.positional_arguments.empty()) {
    for (CLIPositionalArgumentSpec const &pos_arg_spec : cli.positional_arguments) {
      oss << "  ";
      if (pos_arg_spec.options.has_value()) {
        oss << pos_arg_spec.options.value();
      } else {
        oss << pos_arg_spec.name;
      }
      oss << std::endl;
    }

  }
  oss << std::endl;
  oss << "options:" << std::endl;
  oss << "  -h, --help         show this help message and exit" << std::endl;
  for (CLIFlagSpec const &flag_spec : cli.flags) {
    oss << "  ";
    if (flag_spec.short_flag.has_value()) {
      oss << "-" << flag_spec.short_flag.value() << ", ";
    }
    oss << "--" << flag_spec.long_flag << std::endl;
  }

  return oss.str();
}

} // namespace FlexFlow
