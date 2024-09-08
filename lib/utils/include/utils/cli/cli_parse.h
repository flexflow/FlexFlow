#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_PARSE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_PARSE_H

#include "utils/cli/cli_parse_result.dtg.h"
#include "utils/cli/cli_spec.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

tl::expected<CLIFlagKey, std::string> cli_parse_flag(CLISpec const &cli,
                                                     std::string const &arg);
tl::expected<CLIParseResult, std::string>
    cli_parse(CLISpec const &, std::vector<std::string> const &);
tl::expected<CLIParseResult, std::string>
    cli_parse(CLISpec const &, int argc, char const *const *argv);

} // namespace FlexFlow

#endif
