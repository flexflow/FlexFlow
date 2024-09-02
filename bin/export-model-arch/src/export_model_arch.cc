#include "pcg/file_format/v1/v1_computation_graph.h"
#include "utils/cli/cli_spec.h"
#include "utils/exception.h"
#include "utils/cli/cli_parse_result.h"
#include "models/transformer.h"

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

  TransformerConfig config = TransformerConfig{/*num_features=*/512,
                                               /*sequence_length=*/512,
                                               /*batch_size=*/64,
                                               /*dim_feedforward=*/2048,
                                               /*num_heads=*/8,
                                               /*num_encoder_layers=*/6,
                                               /*num_decoder_layers=*/6,
                                               /*dropout=*/0.1,
                                               /*layer_norm_eps=*/1e-05,
                                               /*vocab_size=*/64};
  ComputationGraph cg = get_transformer_computation_graph(config);

  nlohmann::json j = to_v1(cg);

  std::cout << j.dump(2) << std::endl;
}
