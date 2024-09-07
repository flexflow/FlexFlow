#include "utils/cli/cli_get_help_message.h"
#include "utils/join_strings.h"
#include <sstream>

namespace FlexFlow {

std::string cli_get_help_message(std::string const &program_name,
                                 CLISpec const &cli) {
  std::ostringstream oss;

  oss << "usage: " << program_name;
  oss << " [-h]";
  for (CLIFlagSpec const &flag_spec : cli.flags) {
    oss << " [--" << flag_spec.long_flag << "]";
  }
  for (CLIPositionalArgumentSpec const &pos_arg_spec :
       cli.positional_arguments) {
    if (pos_arg_spec.options.has_value()) {
      oss << " {" << join_strings(pos_arg_spec.options.value(), ",") << "}";
    } else {
      oss << " " << pos_arg_spec.name;
    }
  }

  oss << std::endl;
  oss << std::endl;
  oss << "positional arguments:" << std::endl;
  if (!cli.positional_arguments.empty()) {
    for (CLIPositionalArgumentSpec const &pos_arg_spec :
         cli.positional_arguments) {
      oss << "  ";
      if (pos_arg_spec.options.has_value()) {
        oss << "{" << join_strings(pos_arg_spec.options.value(), ",") << "}";
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
