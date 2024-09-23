#include "utils/cli/cli_spec.h"
#include "utils/containers/count.h"
#include "utils/containers/transform.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

CLISpec empty_cli_spec() {
  return CLISpec{{}, {}};
}

std::vector<CLIFlagKey> cli_get_flag_keys(CLISpec const &cli) {
  return transform(count(cli.flags.size()),
                   [](int idx) { return CLIFlagKey{idx}; });
}

CLIArgumentKey cli_add_help_flag(CLISpec &cli) {
  CLIFlagSpec help_flag =
      CLIFlagSpec{"help", 'h', "show this help message and exit"};
  return cli_add_flag(cli, help_flag);
}

CLIArgumentKey cli_add_flag(CLISpec &cli, CLIFlagSpec const &flag_spec) {
  cli.flags.push_back(flag_spec);

  return CLIArgumentKey{CLIFlagKey{int_from_size_t(cli.flags.size()) - 1}};
}

CLIArgumentKey
    cli_add_positional_argument(CLISpec &cli,
                                CLIPositionalArgumentSpec const &arg) {
  cli.positional_arguments.push_back(arg);
  return CLIArgumentKey{CLIPositionalArgumentKey{
      int_from_size_t(cli.positional_arguments.size()) - 1}};
}

} // namespace FlexFlow
