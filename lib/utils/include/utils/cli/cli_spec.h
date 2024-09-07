#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_SPEC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_SPEC_H

#include "utils/cli/cli_argument_key.dtg.h"
#include "utils/cli/cli_flag_spec.dtg.h"
#include "utils/cli/cli_parse_result.dtg.h"
#include "utils/cli/cli_spec.dtg.h"
#include <tl/expected.hpp>
#include <unordered_set>

namespace FlexFlow {

CLISpec empty_cli_spec();
std::unordered_set<std::string> cli_get_all_keys(CLISpec const &);
bool cli_has_key(CLISpec const &, std::string const &);
CLIArgumentKey cli_add_help_flag(CLISpec &);
CLIArgumentKey cli_add_flag(CLISpec &, CLIFlagSpec const &);
CLIArgumentKey cli_add_positional_argument(CLISpec &,
                                           CLIPositionalArgumentSpec const &);
tl::expected<CLIParseResult, std::string>
    cli_parse(CLISpec const &, std::vector<std::string> const &);
tl::expected<CLIParseResult, std::string>
    cli_parse(CLISpec const &, int argc, char const *const *argv);

} // namespace FlexFlow

#endif
