#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_SPEC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_SPEC_H

#include "utils/cli/cli_argument_key.dtg.h"
#include "utils/cli/cli_flag_spec.dtg.h"
#include "utils/cli/cli_spec.dtg.h"
#include <unordered_set>

namespace FlexFlow {

CLISpec empty_cli_spec();
std::vector<CLIFlagKey> cli_get_flag_keys(CLISpec const &);
CLIArgumentKey cli_add_help_flag(CLISpec &);
CLIArgumentKey cli_add_flag(CLISpec &, CLIFlagSpec const &);
CLIArgumentKey cli_add_positional_argument(CLISpec &,
                                           CLIPositionalArgumentSpec const &);

} // namespace FlexFlow

#endif
