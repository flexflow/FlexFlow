#include "pcg/file_format/v1/v1_mapped_parallel_computation_graph.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.dtg.h"
#include "realm-execution/distributed_ff_handle.h"
#include "realm-execution/pcg_instance.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/realm_manager.h"
#include "utils/cli/cli_get_help_message.h"
#include "utils/cli/cli_parse.h"
#include "utils/cli/cli_parse_result.h"
#include "utils/cli/cli_spec.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/positive_int/positive_int.h"
#include <fstream>
#include <string_view>

using namespace FlexFlow;

static char *leak_string_contents(std::string_view str) {
  // Realm command-line arguments require char* so intentionally leak the
  // allocated string contents here
  std::vector<char> *content = new std::vector<char>{str.begin(), str.end()};
  content->push_back(0); // NUL byte
  return content->data();
}

static std::vector<char *> make_realm_args(std::string_view executable_name) {
  std::vector<char *> result;
  result.push_back(leak_string_contents(executable_name));
  return result;
}

int main(int argc, char **argv) {
  CLISpec cli;

  CLIArgumentKey arg_key_help = cli_add_help_flag(cli);

  CLIArgumentKey key_mapped_pcg_json =
      cli.add_positional_argument(CLIPositionalArgumentSpec{
          /*name=*/"mapped_pcg_json",
          /*choices=*/std::nullopt,
          /*description=*/
          "path to a file containing mappped PCG encoded as JSON",
      });

  ASSERT(argc >= 1);
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

  std::string mapped_pcg_json =
      cli_get_positional_argument(parsed, key_mapped_pcg_json);

  std::vector<char *> realm_args = make_realm_args(prog_name);
  int realm_argc = realm_args.size();
  char **realm_argv = realm_args.data();
  RealmManager manager(&realm_argc, &realm_argv);

  ControllerTaskResult result =
      manager.start_controller([&](RealmContext &ctx) {
        MappedParallelComputationGraph mpcg = [&]() {
          std::ifstream f(mapped_pcg_json);
          nlohmann::json mpcg_json = nlohmann::json::parse(f);
          return from_v1(mpcg_json.get<V1MappedParallelComputationGraph>());
        }();

        // instantiate computation graph
        OptimizerAttrs optimizer_attrs =
            OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                             /*momentum=*/0.9,
                                             /*nesterov=*/false,
                                             /*weight_decay=*/0.001}};

        std::map<DynamicValueAttrs, DynamicTensorAccessor> input_tensors;

        DistributedFfHandle device_handle =
            create_distributed_ff_handle(ctx,
                                         /*workSpaceSize=*/1024 * 1024,
                                         /*allowTensorOpMathConversion=*/true);

        bool has_gpus = []() {
          FlexFlow::Realm::Machine::ProcessorQuery pq(
              FlexFlow::Realm::Machine::get_machine());
          pq.only_kind(FlexFlow::Realm::Processor::Kind::TOC_PROC);
          return pq.count() > 0;
        }();

        PCGInstance pcg_instance = create_pcg_instance(
            /*ctx=*/ctx,
            /*mpcg=*/mpcg,
            /*optimizer=*/optimizer_attrs,
            /*loss=*/std::nullopt,
            /*input_tensors=*/input_tensors,
            /*profiling_settings=*/ProfilingSettings{0_n, 1_p},
            /*device_handle=*/device_handle,
            /*device_type=*/has_gpus ? DeviceType::GPU : DeviceType::CPU);

        // begin training loop
        int num_epochs = 5;
        for (int i = 0; i < num_epochs; i++) {
          perform_all_passes_for_pcg_instance(
              /*instance=*/pcg_instance,
              /*profiling_settings=*/ProfilingSettings{0_n, 1_p},
              /*device_handle=*/device_handle);
        }
      });
  result.wait();

  return 0;
}
