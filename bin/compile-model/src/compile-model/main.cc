#include "compiler/compiler.h"
#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/data_parallelism/data_parallelism_config.dtg.h"
#include "compiler/mcmc/mcmc_over_mapped_pcg_config.dtg.h"
#include "compiler/search_result.h"
#include "compiler/unity_algorithm/unity_search_config.dtg.h"
#include "kernels/allocation.h"
#include "kernels/device_handle_t.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "local-execution/cost_estimator/local_cost_estimator.h"
#include "pcg/file_format/v1/v1_computation_graph.h"
#include "pcg/file_format/v1/v1_machine_specification.dtg.h"
#include "pcg/file_format/v1/v1_machine_specification.h"
#include "pcg/file_format/v1/v1_mapped_parallel_computation_graph.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "pcg/pcg_from_computation_graph.h"
#include "utils/cli/cli_get_help_message.h"
#include "utils/cli/cli_parse.h"
#include "utils/cli/cli_parse_result.h"
#include "utils/cli/cli_spec.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/not_implemented.h"
#include "utils/optional.h"
#include "utils/positive_int/positive_int.h"
#include <fstream>
#include <optional>

using namespace FlexFlow;

static std::pair<parallel_layer_guid_t, MappedOperatorTaskGroup>
    single_device_mapping_from_pcg_invocation_info(
        ParallelLayerInvocationInfo const &info) {
  // Everything maps to zero
  ParallelTensorSpaceCoordinate tensor_coord_zero{
      0_n, 0_n, FFOrdered<nonnegative_int>{0_n}};
  MachineSpaceCoordinate machine_coord_zero{0_n, 0_n};
  OperatorAtomicTaskShardBinding shard_binding{binary_merge_disjoint_maps(
      map_values(info.incoming,
                 [&](ParallelTensorInfo const &) { return tensor_coord_zero; }),
      map_values(info.outgoing, [&](ParallelTensorInfo const &) {
        return tensor_coord_zero;
      }))};
  return std::pair<parallel_layer_guid_t, MappedOperatorTaskGroup>{
      info.layer_info.guid,
      MappedOperatorTaskGroup{{{machine_coord_zero, shard_binding}}},
  };
}

static MappedParallelComputationGraph
    lift_cg_to_mpcg_for_single_device(ComputationGraph const &cg) {
  ParallelComputationGraph pcg = pcg_from_computation_graph(cg);
  std::map<parallel_layer_guid_t, MappedOperatorTaskGroup>
      mapped_op_task_groups = map_from_pairs(
          transform(pcg_get_invocation_info_set(pcg),
                    single_device_mapping_from_pcg_invocation_info));
  return mapped_pcg_from_pcg_and_mapped_op_task_groups(pcg,
                                                       mapped_op_task_groups);
}

static MachineSpecification discover_machine() {
  // TODO(Elliott): actually discover the machine topology
  NOT_IMPLEMENTED();
}

static Allocator create_allocator(bool cpu_only) {
  if (cpu_only) {
    return create_local_cpu_memory_allocator();
  } else {
    return create_local_cuda_memory_allocator();
  }
}

static std::optional<ManagedPerDeviceFFHandle>
    create_device_handle(bool cpu_only) {
  if (cpu_only) {
    return std::nullopt;
  }

  return initialize_single_gpu_handle(
      /*workSpaceSize=*/1024 * 1024,
      /*allowTensorOpMathConversion=*/true);
}

static device_handle_t create_device_handle(
    bool cpu_only,
    std::optional<ManagedPerDeviceFFHandle> const &managed_handle) {
  if (cpu_only) {
    return cpu_make_device_handle_t();
  } else {
    return gpu_make_device_handle_t(assert_unwrap(managed_handle).raw_handle());
  }
}

static CostEstimator create_cost_estimator(
    MachineSpecification const &machine,
    bool cpu_only,
    std::optional<ManagedPerDeviceFFHandle> const &managed_handle) {
  Allocator allocator = create_allocator(cpu_only);
  device_handle_t ff_handle = create_device_handle(cpu_only, managed_handle);
  global_device_id_t global_device_id = global_device_id_t{
      /*coord=*/MachineSpaceCoordinate{
          /*node_idx=*/0_n,
          /*device_idx=*/0_n,
      },
      /*device_type=*/(cpu_only ? DeviceType::CPU : DeviceType::GPU),
  };
  return CostEstimator::create<LocalCostEstimator>(
      machine.interconnect_specification,
      allocator,
      ProfilingSettings{/*warmup_iters=*/2_n, /*measure_iters=*/5_p},
      ff_handle,
      global_device_id);
}

static AlgorithmConfig
    select_compiler_algorithm(std::string const &strategy,
                              bool cpu_only,
                              MachineSpecification const &machine) {
  if (strategy == "data_parallel") {
    positive_int degree =
        machine.compute_specification.num_nodes *
        (cpu_only ? machine.compute_specification.num_cpus_per_node
                  : machine.compute_specification.num_gpus_per_node);
    return AlgorithmConfig{
        DataParallelismConfig{/*degree=*/degree.int_from_positive_int()}};
  } else if (strategy == "unity") {
    // TODO: pick better defaults
    return AlgorithmConfig{
        UnitySearchConfig{/*alpha=*/0.5, /*budget=*/100, /*max_num_ops=*/100}};
  } else if (strategy == "mcmc") {
    // TODO: pick better defaults
    return AlgorithmConfig{
        MCMCOverMappedPCGConfig{/*temperature=*/0.5,
                                /*num_iterations=*/100_n,
                                /*substitution_frequency=*/0.5}};
  } else {
    PANIC("no such strategy: {}", strategy);
  }
}

static MachineSpecification get_machine_specification(
    std::optional<std::string> const &machine_specification_json_path) {
  if (machine_specification_json_path.has_value()) {
    return [&]() {
      std::ifstream f{machine_specification_json_path.value()};
      nlohmann::json machine_specification_json = nlohmann::json::parse(f);
      return from_v1(machine_specification_json.get<V1MachineSpecification>());
    }();
  } else {
    return discover_machine();
  }
}

int main(int argc, char **argv) {
  CLISpec cli;

  CLIArgumentKey arg_key_help = cli_add_help_flag(cli);

  CLIArgumentKey key_cg_json =
      cli.add_positional_argument(CLIPositionalArgumentSpec{
          /*name=*/"cg-json",
          /*choicse=*/std::nullopt,
          /*description=*/
          "path to a file containing computation graph, encoded as JSON",
      });

  CLIArgumentKey key_mpcg_json_output =
      cli.add_positional_argument(CLIPositionalArgumentSpec{
          /*name=*/"mpcg-json-output",
          /*choices=*/std::nullopt,
          /*description=*/
          "path to write the resulting mapping PCG, encoded as JSON",
      });

  std::vector<std::string> strategy_options = {
      "passthrough", "data-parallel", "unity", "mcmc"};

  CLIArgumentKey key_strategy =
      cli.add_positional_argument(CLIPositionalArgumentSpec{
          /*name=*/"strategy",
          /*choices=*/strategy_options,
          /*description=*/"compilation strategy for building the mapped PCG",
      });

  CLIArgumentKey key_cpu = cli.add_flag(CLIFlagSpec{
      /*long_flag=*/"cpu",
      /*short_flag=*/std::nullopt,
      /*description=*/"optimize graph for CPUs only (no GPUs)",
  });

  CLIArgumentKey key_machine_specification_file =
      cli.add_named_argument(CLINamedArgumentSpec{
          /*long_flag=*/"machine-spec-json",
          /*metavar=*/"PATH",
          /*choices=*/std::nullopt,
          /*specification=*/
          ("specification of the target machine, encoded as json. "
           "if none is provided, the current machine will be assumed to be the "
           "target machine."),
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

  std::string cg_json = cli_get_positional_argument(parsed, key_cg_json);
  std::string mpcg_json_output =
      cli_get_positional_argument(parsed, key_mpcg_json_output);
  std::string strategy = cli_get_positional_argument(parsed, key_strategy);
  std::optional<std::string> machine_specification_json_path =
      cli_get_named_argument(parsed, key_machine_specification_file);
  bool cpu = cli_get_flag(parsed, key_cpu);

  ComputationGraph cg = [&]() {
    std::ifstream f{cg_json};
    nlohmann::json cg_json = nlohmann::json::parse(f);
    return from_v1(cg_json.get<V1ComputationGraph>());
  }();

  MappedParallelComputationGraph mpcg = [&]() {
    if (strategy == "passthrough") {
      return lift_cg_to_mpcg_for_single_device(cg);
    } else {
      // Need to root this on the stack so it stays alive for the whole session
      std::optional<ManagedPerDeviceFFHandle> managed_handle =
          create_device_handle(cpu);
      MachineSpecification machine_specification =
          get_machine_specification(machine_specification_json_path);
      CostEstimator estimator =
          create_cost_estimator(machine_specification, cpu, managed_handle);
      AlgorithmConfig algorithm =
          select_compiler_algorithm(strategy, cpu, machine_specification);
      SearchResult result =
          optimize(cg, machine_specification, estimator, algorithm);
      return get_mapped_pcg_from_search_result(result);
    }
  }();

  {
    std::ofstream f{mpcg_json_output};
    nlohmann::json mpcg_json = to_v1(mpcg);
    f << mpcg_json;
  }

  return 0;
}
