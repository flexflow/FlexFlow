#include "compiler/series_parallel/computation_graph_binary_sp_decomposition.h"
#include "compiler/series_parallel/get_computation_graph_series_parallel_decomposition.h"
#include "export_model_arch/json_sp_model_export.dtg.h"
#include "models/inception_v3/inception_v3.h"
#include "models/split_test/split_test.h"
#include "models/transformer/transformer.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "pcg/computation_graph.h"
#include "pcg/file_format/v1/v1_computation_graph.h"
#include "utils/cli/cli_get_help_message.h"
#include "utils/cli/cli_parse.h"
#include "utils/cli/cli_parse_result.h"
#include "utils/cli/cli_spec.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/transform.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/right_associative_binary_sp_tree_from_nary.h"
#include "utils/graph/series_parallel/get_series_parallel_decomposition.h"

using namespace ::FlexFlow;

ComputationGraph get_single_operator_computation_graph() {
  ComputationGraphBuilder b;

  size_t batch_size = 8;
  size_t in_channels = 16;
  size_t out_channels = 12;
  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
          batch_size,
          in_channels,
          out_channels,
      }},
      DataType::FLOAT,
  };

  tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);

  InitializerAttrs kernel_initializer =
      InitializerAttrs{GlorotUniformAttrs{/*seed=*/12}};
  InitializerAttrs bias_initializer = InitializerAttrs{ZeroInitializerAttrs{}};
  tensor_guid_t output = b.dense(input,
                                 in_channels,
                                 Activation::RELU,
                                 /*use_bias=*/true,
                                 DataType::FLOAT,
                                 kernel_initializer,
                                 bias_initializer,
                                 "my_example_operator");

  return b.computation_graph;
}

ComputationGraph get_default_transformer_computation_graph() {
  TransformerConfig config = get_default_transformer_config();
  ComputationGraph cg = get_transformer_computation_graph(config);

  return cg;
}

tl::expected<ComputationGraph, std::string>
    get_model_computation_graph(std::string const &model_name) {
  if (model_name == "transformer") {
    return get_default_transformer_computation_graph();
  } else if (model_name == "inception_v3") {
    return get_inception_v3_computation_graph(
      get_default_inception_v3_training_config());
  } else if (model_name == "split_test") {
    int batch_size = 8;
    return get_split_test_computation_graph(batch_size);
  } else if (model_name == "single_operator") {
    return get_single_operator_computation_graph();
  } else {
    return tl::unexpected(fmt::format("Unknown model name: {}", model_name));
  }
}

tl::expected<JsonSPModelExport, std::string>
    get_sp_model_export(std::string const &model_name) {
  ComputationGraph computation_graph = ({
    tl::expected<ComputationGraph, std::string> result =
        get_model_computation_graph(model_name);
    if (!result.has_value()) {
      return tl::unexpected(result.error());
    }
    result.value();
  });

  ComputationGraphBinarySPDecomposition sp_decomposition = ({
    std::optional<ComputationGraphBinarySPDecomposition> result =
        get_computation_graph_right_assoc_binary_sp_decomposition(
            computation_graph);
    if (!result.has_value()) {
      return tl::unexpected("Failed to generate series-parallel decomposition "
                            "of computation graph.");
    }
    result.value();
  });

  std::pair<V1ComputationGraph, bidict<int, layer_guid_t>> v1_result =
      to_v1_including_node_numbering(computation_graph);
  V1ComputationGraph v1_cg = v1_result.first;
  bidict<int, layer_guid_t> layer_numbering = v1_result.second;
  GenericBinarySPDecompositionTree<int> v1_sp_decomposition =
      transform(sp_decomposition.raw_tree,
                [&](layer_guid_t const &l) { return layer_numbering.at_r(l); });

  return JsonSPModelExport{
      v1_sp_decomposition,
      v1_cg,
  };
}

int main(int argc, char **argv) {
  CLISpec cli = empty_cli_spec();

  CLIArgumentKey arg_key_help = cli_add_help_flag(cli);

  CLIArgumentKey key_sp_decomposition =
      cli_add_flag(cli,
                   CLIFlagSpec{"sp-decomposition",
                               std::nullopt,
                               "also output a series parallel decomposition of "
                               "the model's computation graph"});

  CLIArgumentKey key_dot = cli_add_flag(
      cli,
      CLIFlagSpec{
          "dot",
          std::nullopt,
          "output a dot representation of the model's computation graph"});

  CLIArgumentKey key_preprocessed_dot = cli_add_flag(
      cli,
      CLIFlagSpec{"preprocessed-dot",
                  std::nullopt,
                  "output a dot representation of model's computation graph "
                  "for preprocessed to help check series-parallel structure"});

  std::vector<std::string> model_options = {
      "transformer", "inception_v3", "split_test", "single_operator"};
  CLIArgumentKey key_model_name = cli_add_positional_argument(
      cli,
      CLIPositionalArgumentSpec{
          "model", model_options, "name of the model to export"});

  assert(argc >= 1);
  std::string prog_name = argv[0];

  CLIParseResult parsed = ({
    tl::expected<CLIParseResult, std::string> result =
        cli_parse(cli, argc, argv);
    if (!result.has_value()) {
      std::string error_msg = result.error();
      std::cerr << cli_get_help_message(prog_name, cli);
      std::cerr << std::endl;
      std::cerr << "error: " << error_msg << std::endl;
      return 1;
    }

    result.value();
  });

  bool help = cli_get_flag(parsed, arg_key_help);
  if (help) {
    std::cerr << cli_get_help_message(prog_name, cli);
    return 1;
  }

  std::string model_name = cli_get_argument(parsed, key_model_name);
  bool sp_decompositition = cli_get_flag(parsed, key_sp_decomposition);
  bool dot = cli_get_flag(parsed, key_dot);
  bool preprocessed_dot = cli_get_flag(parsed, key_preprocessed_dot);

  auto handle_error = [](auto const &result) {
    if (!result.has_value()) {
      std::cerr << "error: " << result.error() << std::endl;
      exit(1);
    }

    return result.value();
  };

  if (dot) {
    ComputationGraph cg = handle_error(get_model_computation_graph(model_name));

    std::cout << as_dot(cg) << std::endl;
    return 0;
  }

  if (preprocessed_dot) {
    ComputationGraph cg = handle_error(get_model_computation_graph(model_name));
    std::string rendered =
        render_preprocessed_computation_graph_for_sp_decomposition(cg);

    std::cout << rendered << std::endl;
    return 0;
  }

  nlohmann::json json_output;
  if (sp_decompositition) {
    JsonSPModelExport model_export =
        handle_error(get_sp_model_export(model_name));

    json_output = model_export;
  } else {
    ComputationGraph cg = handle_error(get_model_computation_graph(model_name));

    json_output = to_v1(cg);
  }
  std::cout << json_output.dump(2) << std::endl;

  return 0;
}
