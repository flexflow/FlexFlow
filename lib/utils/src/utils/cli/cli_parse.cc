#include "utils/cli/cli_parse.h"
#include "utils/containers/enumerate.h"
#include "utils/cli/cli_spec.h"
#include "utils/containers/contains.h"
#include "utils/containers/generate_map.h"

namespace FlexFlow {

tl::expected<CLIFlagKey, std::string> cli_parse_flag(CLISpec const &cli,
                                                     std::string const &arg) {
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

  return tl::unexpected(fmt::format("Encountered unknown flag {}", arg));
}


tl::expected<CLIParseResult, std::string>
    cli_parse(CLISpec const &cli, std::vector<std::string> const &args) {
  CLIParseResult result = CLIParseResult{
      generate_map(cli_get_flag_keys(cli),
                   [](CLIFlagKey const &) { return false; }),
      {},
  };

  int consumed_positional_args = 0;
  auto parse_positional_arg =
      [&](std::string const &arg) -> std::optional<std::string> {
    if (consumed_positional_args >= cli.positional_arguments.size()) {
      return fmt::format("Too many positional arguments: expected {}",
                         cli.positional_arguments.size());
    }

    CLIPositionalArgumentSpec arg_spec =
        cli.positional_arguments.at(consumed_positional_args);

    if (arg_spec.choices.has_value() &&
        !contains(arg_spec.choices.value(), arg)) {
      return fmt::format("Invalid option for positional argument \"{}\": \"{}\"", arg_spec.name, arg);
    }

    result.positional_arguments.insert(
        {CLIPositionalArgumentKey{consumed_positional_args}, arg});
    consumed_positional_args++;

    return std::nullopt;
  };

  for (int i = 1; i < args.size(); i++) {
    std::string arg = args.at(i);

    if (!arg.empty() && arg.at(0) == '-') {
      tl::expected<CLIFlagKey, std::string> parsed_flag =
          cli_parse_flag(cli, arg);

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
    return tl::unexpected(
        fmt::format("Not enough positional arguments: found {}, expected {}",
                    consumed_positional_args,
                    cli.positional_arguments.size()));
  }

  return result;
}

tl::expected<CLIParseResult, std::string>
    cli_parse(CLISpec const &cli, int argc, char const *const *argv) {
  std::vector<std::string> args = {argv, argv + argc};

  return cli_parse(cli, args);
}


} // namespace FlexFlow
