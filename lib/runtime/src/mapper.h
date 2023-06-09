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

#include "legion.h"
#include "model.h"
#include "null_mapper.h"
#include "pcg/device_id.h"
#include "pcg/device_type.h"
#include "tasks.h"

namespace FlexFlow {

class FFShardingFunctor : public Legion::ShardingFunctor {
public:
  FFShardingFunctor(int gpus_per_node,
                    int cpus_per_node,
                    int num_nodes,
                    MachineView const &mv);

public:
  Legion::ShardID shard(Legion::DomainPoint const &point,
                        Legion::Domain const &full_space,
                        const size_t total_shards);

private:
  Legion::ShardID get_shard_id(device_id_t) const;

private:
  int gpus_per_node, cpus_per_node, num_nodes;
  MachineView machine_view;
};

struct InstanceCreationLog {
  std::string task_name;
  size_t size;
  Legion::Memory memory;
  Legion::Processor processor;
};

void register_sharding_functor(Legion::Runtime *, FFShardingFunctor *);
void register_sharding_functor(Legion::Runtime *,
                               std::size_t,
                               FFShardingFunctor *);
std::vector<MachineView> starting_at_all_devices(MachineView const &,
                                                 int total_num_cpus,
                                                 int total_num_gpus);
device_id_t get_device_index(MachineView const &,
                             Legion::DomainPoint const &,
                             Legion::Domain const &);

struct NodesConfig {
  NodesConfig(int num_nodes, int cpus_per_node, int gpus_per_node);

  int get_cpus_per_node() const;
  int get_gpus_per_node() const;
  int get_num_nodes() const;
  int get_total_num_cpus() const;
  int get_total_num_gpus() const;

  FFShardingFunctor *make_sharding_functor(MachineView const &) const;
  std::vector<MachineView> starting_at_all_devices(MachineView const &);

private:
  int cpus_per_node, gpus_per_node, num_nodes;
};

class FFMapper : public Legion::Mapping::NullMapper {
public:
  FFMapper(Legion::Mapping::MapperRuntime *rt,
           Legion::Machine machine,
           Legion::Processor local,
           char const *mapper_name, // const std::string& strategyFile,
           bool _enable_control_replication,
           bool _log_instance_creation);
  ~FFMapper();
  virtual char const *get_mapper_name(void) const;
  virtual MapperSyncModel get_mapper_sync_model(void) const;

public:
  static void update_mappers(Legion::Machine machine,
                             Legion::Runtime *rt,
                             std::set<Legion::Processor> const &local_procs);
  static void register_sharding_functors(Legion::Runtime *runtime,
                                         Legion::Machine machine,
                                         int argv,
                                         char **argc);
  virtual void select_task_options(const Legion::Mapping::MapperContext ctx,
                                   Legion::Task const &task,
                                   TaskOptions &output);
  virtual void premap_task(const Legion::Mapping::MapperContext ctx,
                           Legion::Task const &task,
                           PremapTaskInput const &input,
                           PremapTaskOutput &output);
  virtual void slice_task(const Legion::Mapping::MapperContext ctx,
                          Legion::Task const &task,
                          SliceTaskInput const &input,
                          SliceTaskOutput &output);
  virtual void map_task(const Legion::Mapping::MapperContext ctx,
                        Legion::Task const &task,
                        MapTaskInput const &input,
                        MapTaskOutput &output);
  virtual void map_replicate_task(const Legion::Mapping::MapperContext ctx,
                                  Legion::Task const &task,
                                  MapTaskInput const &input,
                                  MapTaskOutput const &default_output,
                                  MapReplicateTaskOutput &output);
  virtual void select_task_variant(const Legion::Mapping::MapperContext ctx,
                                   Legion::Task const &task,
                                   SelectVariantInput const &input,
                                   SelectVariantOutput &output);
  virtual void postmap_task(const Legion::Mapping::MapperContext ctx,
                            Legion::Task const &task,
                            PostMapInput const &input,
                            PostMapOutput &output);
  virtual void select_task_sources(const Legion::Mapping::MapperContext ctx,
                                   Legion::Task const &task,
                                   SelectTaskSrcInput const &input,
                                   SelectTaskSrcOutput &output);
  virtual void
      create_task_temporary_instance(const Legion::Mapping::MapperContext ctx,
                                     Legion::Task const &task,
                                     CreateTaskTemporaryInput const &input,
                                     CreateTaskTemporaryOutput &output);
  virtual void speculate(const Legion::Mapping::MapperContext ctx,
                         Legion::Task const &task,
                         SpeculativeOutput &output);
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                Legion::Task const &task,
                                TaskProfilingInfo const &input);
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       Legion::Task const &task,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Inline mapping calls
  virtual void map_inline(const Legion::Mapping::MapperContext ctx,
                          Legion::InlineMapping const &inline_op,
                          MapInlineInput const &input,
                          MapInlineOutput &output);
  virtual void select_inline_sources(const Legion::Mapping::MapperContext ctx,
                                     Legion::InlineMapping const &inline_op,
                                     SelectInlineSrcInput const &input,
                                     SelectInlineSrcOutput &output);
  virtual void
      create_inline_temporary_instance(const Legion::Mapping::MapperContext ctx,
                                       Legion::InlineMapping const &inline_op,
                                       CreateInlineTemporaryInput const &input,
                                       CreateInlineTemporaryOutput &output);
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                Legion::InlineMapping const &inline_op,
                                InlineProfilingInfo const &input);

public: // Copy mapping calls
  virtual void map_copy(const Legion::Mapping::MapperContext ctx,
                        Legion::Copy const &copy,
                        MapCopyInput const &input,
                        MapCopyOutput &output);
  virtual void select_copy_sources(const Legion::Mapping::MapperContext ctx,
                                   Legion::Copy const &copy,
                                   SelectCopySrcInput const &input,
                                   SelectCopySrcOutput &output);
  virtual void
      create_copy_temporary_instance(const Legion::Mapping::MapperContext ctx,
                                     Legion::Copy const &copy,
                                     CreateCopyTemporaryInput const &input,
                                     CreateCopyTemporaryOutput &output);
  virtual void speculate(const Legion::Mapping::MapperContext ctx,
                         Legion::Copy const &copy,
                         SpeculativeOutput &output);
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                Legion::Copy const &copy,
                                CopyProfilingInfo const &input);
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       Legion::Copy const &copy,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Close mapping calls
  virtual void map_close(const Legion::Mapping::MapperContext ctx,
                         Legion::Close const &close,
                         MapCloseInput const &input,
                         MapCloseOutput &output);
  virtual void select_close_sources(const Legion::Mapping::MapperContext ctx,
                                    Legion::Close const &close,
                                    SelectCloseSrcInput const &input,
                                    SelectCloseSrcOutput &output);
  virtual void
      create_close_temporary_instance(const Legion::Mapping::MapperContext ctx,
                                      Legion::Close const &close,
                                      CreateCloseTemporaryInput const &input,
                                      CreateCloseTemporaryOutput &output);
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                Legion::Close const &close,
                                CloseProfilingInfo const &input);
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       Legion::Close const &close,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Acquire mapping calls
  virtual void map_acquire(const Legion::Mapping::MapperContext ctx,
                           Legion::Acquire const &acquire,
                           MapAcquireInput const &input,
                           MapAcquireOutput &output);
  virtual void speculate(const Legion::Mapping::MapperContext ctx,
                         Legion::Acquire const &acquire,
                         SpeculativeOutput &output);
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                Legion::Acquire const &acquire,
                                AcquireProfilingInfo const &input);
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       Legion::Acquire const &acquire,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Release mapping calls
  virtual void map_release(const Legion::Mapping::MapperContext ctx,
                           Legion::Release const &release,
                           MapReleaseInput const &input,
                           MapReleaseOutput &output);
  virtual void select_release_sources(const Legion::Mapping::MapperContext ctx,
                                      Legion::Release const &release,
                                      SelectReleaseSrcInput const &input,
                                      SelectReleaseSrcOutput &output);
  virtual void create_release_temporary_instance(
      const Legion::Mapping::MapperContext ctx,
      Legion::Release const &release,
      CreateReleaseTemporaryInput const &input,
      CreateReleaseTemporaryOutput &output);
  virtual void speculate(const Legion::Mapping::MapperContext ctx,
                         Legion::Release const &release,
                         SpeculativeOutput &output);
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                Legion::Release const &release,
                                ReleaseProfilingInfo const &input);
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       Legion::Release const &release,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Partition mapping calls
  virtual void
      select_partition_projection(const Legion::Mapping::MapperContext ctx,
                                  Legion::Partition const &partition,
                                  SelectPartitionProjectionInput const &input,
                                  SelectPartitionProjectionOutput &output);
  virtual void map_partition(const Legion::Mapping::MapperContext ctx,
                             Legion::Partition const &partition,
                             MapPartitionInput const &input,
                             MapPartitionOutput &output);
  virtual void
      select_partition_sources(const Legion::Mapping::MapperContext ctx,
                               Legion::Partition const &partition,
                               SelectPartitionSrcInput const &input,
                               SelectPartitionSrcOutput &output);
  virtual void create_partition_temporary_instance(
      const Legion::Mapping::MapperContext ctx,
      Legion::Partition const &partition,
      CreatePartitionTemporaryInput const &input,
      CreatePartitionTemporaryOutput &output);
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                Legion::Partition const &partition,
                                PartitionProfilingInfo const &input);
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       Legion::Partition const &partition,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Fill mapper calls
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       Legion::Fill const &fill,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public: // Task execution mapping calls
  virtual void configure_context(const Legion::Mapping::MapperContext ctx,
                                 Legion::Task const &task,
                                 ContextConfigOutput &output);
  virtual void select_tunable_value(const Legion::Mapping::MapperContext ctx,
                                    Legion::Task const &task,
                                    SelectTunableInput const &input,
                                    SelectTunableOutput &output);

public: // Must epoch mapping
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       Legion::MustEpoch const &epoch,
                                       SelectShardingFunctorInput const &input,
                                       MustEpochShardingFunctorOutput &output);
  virtual void map_must_epoch(const Legion::Mapping::MapperContext ctx,
                              MapMustEpochInput const &input,
                              MapMustEpochOutput &output);

public: // Dataflow graph mapping
  virtual void map_dataflow_graph(const Legion::Mapping::MapperContext ctx,
                                  MapDataflowGraphInput const &input,
                                  MapDataflowGraphOutput &output);

public: // Memoization control
  virtual void memoize_operation(const Legion::Mapping::MapperContext ctx,
                                 Legion::Mappable const &mappable,
                                 MemoizeInput const &input,
                                 MemoizeOutput &output);

public: // Mapping control and stealing
  virtual void select_tasks_to_map(const Legion::Mapping::MapperContext ctx,
                                   SelectMappingInput const &input,
                                   SelectMappingOutput &output);
  virtual void select_steal_targets(const Legion::Mapping::MapperContext ctx,
                                    SelectStealingInput const &input,
                                    SelectStealingOutput &output);
  virtual void permit_steal_request(const Legion::Mapping::MapperContext ctx,
                                    StealRequestInput const &intput,
                                    StealRequestOutput &output);

private: // static inline methods
  static inline bool physical_sort_func(
      std::pair<Legion::Mapping::PhysicalInstance, unsigned> const &left,
      std::pair<Legion::Mapping::PhysicalInstance, unsigned> const &right) {
    return (left.second < right.second);
  }

private: // Default helper functions
  Legion::Memory
      default_select_target_memory(Legion::Mapping::MapperContext ctx,
                                   Legion::Processor target_proc,
                                   Legion::RegionRequirement const &req);
  bool default_make_instance(Legion::Mapping::MapperContext ctx,
                             Legion::Memory target_mem,
                             Legion::LayoutConstraintSet const &constraints,
                             Legion::Mapping::PhysicalInstance &result,
                             bool meets_constraints,
                             Legion::RegionRequirement const &req,
                             bool &created,
                             size_t *footprint);
  Legion::LayoutConstraintID
      default_select_layout_constraints(Legion::Mapping::MapperContext ctx,
                                        Legion::Memory target_memory,
                                        Legion::RegionRequirement const &req,
                                        bool needs_field_constraint_check);
  void default_select_constraints(Legion::Mapping::MapperContext ctx,
                                  Legion::LayoutConstraintSet &constraints,
                                  Legion::Memory target_memory,
                                  Legion::RegionRequirement const &req);
  void default_policy_select_sources(
      Legion::Mapping::MapperContext ctx,
      Legion::Mapping::PhysicalInstance const &target,
      std::vector<Legion::Mapping::PhysicalInstance> const &sources,
      std::deque<Legion::Mapping::PhysicalInstance> &ranking,
      Legion::Memory preferred_memory = Legion::Memory::NO_MEMORY);

private:
  unsigned long long compute_task_hash(Legion::Task const &task);
  bool is_parameter_server_update_task(Legion::TaskID tid);
  bool is_initializer_task(Legion::TaskID tid);
  std::vector<Legion::Processor> const &
      all_procs_by_kind(Legion::Processor::Kind kind);

  void register_machine_view(Legion::MappingTagID, MachineView const &);
  void register_machine_view(MachineView const &);
  void register_machine_views(std::vector<MachineView> const &);
  std::vector<MachineView> starting_at_all_devices(MachineView const &);

  NodesConfig get_nodes_config() const;

  int get_gpus_per_node() const;
  int get_cpus_per_node() const;

  Legion::Processor const &get_processor(device_id_t);

  bool has_device(device_id_t);

protected:
  Legion::Processor const local_processor;
  Legion::AddressSpace const node_id;
  int total_nodes;
  char const *mapper_name;
  bool enable_control_replication;
  bool log_instance_creation;
  std::vector<Legion::Processor> all_gpus, all_cpus, all_pys, local_gpus,
      local_cpus, local_pys;
  std::map<Legion::Processor, Legion::Memory> proc_fbmems, proc_zcmems;
  std::map<unsigned long long, Legion::Processor> cache_update_tasks;
  // We use MappingTagID has the key since we will pass the tag to the mapper
  std::map<Legion::MappingTagID, MachineView> machine_views;
  std::map<std::pair<Legion::Memory::Kind, Legion::FieldSpace>,
           Legion::LayoutConstraintID>
      layout_constraint_cache;
  std::vector<InstanceCreationLog> created_instances;
};

} // namespace FlexFlow
#endif
