#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_GET_HELP_MESSAGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_GET_HELP_MESSAGE_H

#include "utils/cli/cli_spec.dtg.h"

namespace FlexFlow {

std::string cli_get_help_message(std::string const &program_name,
                                 CLISpec const &);

} // namespace FlexFlow

#endif
