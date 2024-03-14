/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __FLEXFLOW_MAPPER_H__
#define __FLEXFLOW_MAPPER_H__

#include "default_mapper.h"
#include "legion.h"
#include "model.h"
#include "null_mapper.h"

namespace FlexFlow {

using namespace Legion;
using namespace Mapping;

class FFShardingFunctor : public ShardingFunctor {
public:
  FFShardingFunctor(int gpus_per_node,
                    int cpus_per_node,
                    int num_nodes,
                    MachineView const &_mv);
  ~FFShardingFunctor(void);

public:
  ShardID shard(DomainPoint const &point,
                Domain const &full_space,
                const size_t total_shards);

private:
  int gpus_per_node, cpus_per_node, num_nodes;
  MachineView machine_view;
};

struct InstanceCreationLog {
  std::string task_name;
  size_t size;
  Memory memory;
  Processor processor;
};

class FFMapper : public NullMapper {
public:
  FFMapper(MapperRuntime *rt,
           Machine machine,
           Processor local,
           char const *mapper_name, // const std::string& strategyFile,
           bool _enable_control_replication,
           bool _log_instance_creation);
  ~FFMapper();
  virtual char const *get_mapper_name(void) const;
  virtual MapperSyncModel get_mapper_sync_model(void) const;

public:
  static void update_mappers(Machine machine,
                             Runtime *rt,
                             std::set<Processor> const &local_procs);
  static void register_sharding_functor(Runtime *runtime, Machine machine);
  virtual void select_task_options(const MapperContext ctx,
                                   Task const &task,
                                   TaskOptions &output);
  virtual void premap_task(const MapperContext ctx,
                           Task const &task,
                           PremapTaskInput const &input,
                           PremapTaskOutput &output);
  virtual void slice_task(const MapperContext ctx,
                          Task const &task,
                          SliceTaskInput const &input,
                          SliceTaskOutput &output);
  virtual void map_task(const MapperContext ctx,
                        Task const &task,
                        MapTaskInput const &input,
                        MapTaskOutput &output);
  virtual void replicate_task(const MapperContext ctx,
                              Task const &task,
                              ReplicateTaskInput const &input,
                              ReplicateTaskOutput &output);
  virtual void select_task_variant(const MapperContext ctx,
                                   Task const &task,
                                   SelectVariantInput const &input,
                                   SelectVariantOutput &output);
  virtual void postmap_task(const MapperContext ctx,
                            Task const &task,
                            PostMapInput const &input,
                            PostMapOutput &output);
  virtual void select_task_sources(const MapperContext ctx,
                                   Task const &task,
                                   SelectTaskSrcInput const &input,
                                   SelectTaskSrcOutput &output);
  virtual void
      create_task_temporary_instance(const MapperContext ctx,
                                     Task const &task,
                                     CreateTaskTemporaryInput const &input,
                                     CreateTaskTemporaryOutput &output);
  virtual void speculate(const MapperContext ctx,
                         Task const &task,
                         SpeculativeOutput &output);
  virtual void report_profiling(const MapperContext ctx,
                                Task const &task,
                                TaskProfilingInfo const &input);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       Task const &task,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Inline mapping calls
  virtual void map_inline(const MapperContext ctx,
                          InlineMapping const &inline_op,
                          MapInlineInput const &input,
                          MapInlineOutput &output);
  virtual void select_inline_sources(const MapperContext ctx,
                                     InlineMapping const &inline_op,
                                     SelectInlineSrcInput const &input,
                                     SelectInlineSrcOutput &output);
  virtual void
      create_inline_temporary_instance(const MapperContext ctx,
                                       InlineMapping const &inline_op,
                                       CreateInlineTemporaryInput const &input,
                                       CreateInlineTemporaryOutput &output);
  virtual void report_profiling(const MapperContext ctx,
                                InlineMapping const &inline_op,
                                InlineProfilingInfo const &input);

public: // Copy mapping calls
  virtual void map_copy(const MapperContext ctx,
                        Copy const &copy,
                        MapCopyInput const &input,
                        MapCopyOutput &output);
  virtual void select_copy_sources(const MapperContext ctx,
                                   Copy const &copy,
                                   SelectCopySrcInput const &input,
                                   SelectCopySrcOutput &output);
  virtual void
      create_copy_temporary_instance(const MapperContext ctx,
                                     Copy const &copy,
                                     CreateCopyTemporaryInput const &input,
                                     CreateCopyTemporaryOutput &output);
  virtual void speculate(const MapperContext ctx,
                         Copy const &copy,
                         SpeculativeOutput &output);
  virtual void report_profiling(const MapperContext ctx,
                                Copy const &copy,
                                CopyProfilingInfo const &input);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       Copy const &copy,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Close mapping calls
  virtual void map_close(const MapperContext ctx,
                         Close const &close,
                         MapCloseInput const &input,
                         MapCloseOutput &output);
  virtual void select_close_sources(const MapperContext ctx,
                                    Close const &close,
                                    SelectCloseSrcInput const &input,
                                    SelectCloseSrcOutput &output);
  virtual void
      create_close_temporary_instance(const MapperContext ctx,
                                      Close const &close,
                                      CreateCloseTemporaryInput const &input,
                                      CreateCloseTemporaryOutput &output);
  virtual void report_profiling(const MapperContext ctx,
                                Close const &close,
                                CloseProfilingInfo const &input);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       Close const &close,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Acquire mapping calls
  virtual void map_acquire(const MapperContext ctx,
                           Acquire const &acquire,
                           MapAcquireInput const &input,
                           MapAcquireOutput &output);
  virtual void speculate(const MapperContext ctx,
                         Acquire const &acquire,
                         SpeculativeOutput &output);
  virtual void report_profiling(const MapperContext ctx,
                                Acquire const &acquire,
                                AcquireProfilingInfo const &input);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       Acquire const &acquire,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Release mapping calls
  virtual void map_release(const MapperContext ctx,
                           Release const &release,
                           MapReleaseInput const &input,
                           MapReleaseOutput &output);
  virtual void select_release_sources(const MapperContext ctx,
                                      Release const &release,
                                      SelectReleaseSrcInput const &input,
                                      SelectReleaseSrcOutput &output);
  virtual void create_release_temporary_instance(
      const MapperContext ctx,
      Release const &release,
      CreateReleaseTemporaryInput const &input,
      CreateReleaseTemporaryOutput &output);
  virtual void speculate(const MapperContext ctx,
                         Release const &release,
                         SpeculativeOutput &output);
  virtual void report_profiling(const MapperContext ctx,
                                Release const &release,
                                ReleaseProfilingInfo const &input);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       Release const &release,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Partition mapping calls
  virtual void
      select_partition_projection(const MapperContext ctx,
                                  Partition const &partition,
                                  SelectPartitionProjectionInput const &input,
                                  SelectPartitionProjectionOutput &output);
  virtual void map_partition(const MapperContext ctx,
                             Partition const &partition,
                             MapPartitionInput const &input,
                             MapPartitionOutput &output);
  virtual void select_partition_sources(const MapperContext ctx,
                                        Partition const &partition,
                                        SelectPartitionSrcInput const &input,
                                        SelectPartitionSrcOutput &output);
  virtual void create_partition_temporary_instance(
      const MapperContext ctx,
      Partition const &partition,
      CreatePartitionTemporaryInput const &input,
      CreatePartitionTemporaryOutput &output);
  virtual void report_profiling(const MapperContext ctx,
                                Partition const &partition,
                                PartitionProfilingInfo const &input);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       Partition const &partition,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Fill mapper calls
  virtual void select_sharding_functor(const MapperContext ctx,
                                       Fill const &fill,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Task execution mapping calls
  virtual void configure_context(const MapperContext ctx,
                                 Task const &task,
                                 ContextConfigOutput &output);
  virtual void select_tunable_value(const MapperContext ctx,
                                    Task const &task,
                                    SelectTunableInput const &input,
                                    SelectTunableOutput &output);

public: // Must epoch mapping
  virtual void select_sharding_functor(const MapperContext ctx,
                                       MustEpoch const &epoch,
                                       SelectShardingFunctorInput const &input,
                                       MustEpochShardingFunctorOutput &output);
  virtual void map_must_epoch(const MapperContext ctx,
                              MapMustEpochInput const &input,
                              MapMustEpochOutput &output);

public: // Dataflow graph mapping
  virtual void map_dataflow_graph(const MapperContext ctx,
                                  MapDataflowGraphInput const &input,
                                  MapDataflowGraphOutput &output);

public: // Memoization control
  virtual void memoize_operation(const MapperContext ctx,
                                 Mappable const &mappable,
                                 MemoizeInput const &input,
                                 MemoizeOutput &output);

public: // Mapping control and stealing
  virtual void select_tasks_to_map(const MapperContext ctx,
                                   SelectMappingInput const &input,
                                   SelectMappingOutput &output);
  virtual void select_steal_targets(const MapperContext ctx,
                                    SelectStealingInput const &input,
                                    SelectStealingOutput &output);
  virtual void permit_steal_request(const MapperContext ctx,
                                    StealRequestInput const &intput,
                                    StealRequestOutput &output);

private: // static inline methods
  static inline bool
      physical_sort_func(std::pair<PhysicalInstance, unsigned> const &left,
                         std::pair<PhysicalInstance, unsigned> const &right) {
    return (left.second < right.second);
  }

private: // Default helper functions
  Memory default_select_target_memory(MapperContext ctx,
                                      Processor target_proc,
                                      RegionRequirement const &req);
  bool default_make_instance(MapperContext ctx,
                             Memory target_mem,
                             LayoutConstraintSet const &constraints,
                             PhysicalInstance &result,
                             bool meets_constraints,
                             RegionRequirement const &req,
                             bool &created,
                             size_t *footprint);
  LayoutConstraintID
      default_select_layout_constraints(MapperContext ctx,
                                        Memory target_memory,
                                        RegionRequirement const &req,
                                        bool needs_field_constraint_check);
  void default_select_constraints(MapperContext ctx,
                                  LayoutConstraintSet &constraints,
                                  Memory target_memory,
                                  RegionRequirement const &req);
  void default_policy_select_sources(
      MapperContext ctx,
      PhysicalInstance const &target,
      std::vector<PhysicalInstance> const &sources,
      std::deque<PhysicalInstance> &ranking,
      Memory preferred_memory = Memory::NO_MEMORY);

private:
  unsigned long long compute_task_hash(Task const &task);
  bool is_parameter_server_update_task(TaskID tid);
  bool is_initializer_task(TaskID tid);
  std::vector<Processor> const &all_procs_by_kind(Processor::Kind kind);

protected:
  const Processor local_processor;
  const AddressSpace node_id;
  int total_nodes;
  char const *mapper_name;
  bool enable_control_replication;
  bool log_instance_creation;
  std::vector<Processor> all_gpus, all_cpus, all_pys, local_gpus, local_cpus,
      local_pys;
  std::map<Processor, Memory> proc_fbmems, proc_zcmems;
  std::map<unsigned long long, Processor> cache_update_tasks;
  // We use MappingTagID has the key since we will pass the tag to the mapper
  std::map<MappingTagID, MachineView> machine_views;
  std::map<std::pair<Memory::Kind, FieldSpace>, LayoutConstraintID>
      layout_constraint_cache;
  std::vector<InstanceCreationLog> created_instances;
};

}; // namespace FlexFlow
#endif
