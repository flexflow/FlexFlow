#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_PARSE_RESULT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_PARSE_RESULT_H

#include "utils/cli/cli_argument_key.dtg.h"
#include "utils/cli/cli_parse_result.dtg.h"

namespace FlexFlow {

bool cli_get_flag(CLIParseResult const &, CLIArgumentKey const &);
std::string cli_get_argument(CLIParseResult const &, CLIArgumentKey const &);

} // namespace FlexFlow

#endif
