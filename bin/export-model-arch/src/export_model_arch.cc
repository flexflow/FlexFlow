#include "utils/cli/cli_spec.h"
#include "utils/exception.h"
#include "utils/cli/cli_parse_result.h"

using namespace ::FlexFlow;

int main(int argc, char **argv) {
  CLISpec cli = empty_cli_spec();
  CLIArgumentKey key_sp_decomposition = cli_add_flag(cli, CLIFlagSpec{"--sp-decomposition", std::nullopt});
  std::unordered_set<std::string> model_options = {"transformer"};
  CLIArgumentKey key_model_name = cli_add_positional_argument(cli, CLIPositionalArgumentSpec{model_options});

  CLIParseResult parsed = ({
    tl::expected<CLIParseResult, std::string> result = cli_parse(cli, argc, argv);
    if (!result.has_value()) {
      throw mk_runtime_error(result.error());
    }

    result.value();
  });

  bool sp_decompositition = cli_get_flag(parsed, key_sp_decomposition);
  std::string model_name = cli_get_argument(parsed, key_model_name);

  fmt::print("sp_decomposition = {}, model_name = {}", sp_decompositition, model_name);
}
