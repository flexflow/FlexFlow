/* Copyright 2017 Stanford University
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

#ifndef __CNN_MAPPER_H__
#define __CNN_MAPPER_H__

#include "legion.h"
#include "default_mapper.h"
#include "null_mapper.h"
#include "model.h"

using namespace Legion;
using namespace Legion::Mapping;

class FFShardingFunctor : public ShardingFunctor {
public:
  FFShardingFunctor(int gpus_per_node,
                    int cpus_per_node,
                    int num_nodes,
                    ParallelConfig _pc);
  ~FFShardingFunctor(void);
public:
  ShardID shard(const DomainPoint &point,
                const Domain &full_space,
                const size_t total_shards);
private:
  int gpus_per_node, cpus_per_node, num_nodes;
  ParallelConfig config;
};

struct InstanceCreationLog {
  std::string task_name;
  size_t size;
  Memory memory;
  Processor processor;
};

class FFMapper : public NullMapper {
public:
  FFMapper(MapperRuntime *rt, Machine machine, Processor local,
           const char *mapper_name, const std::string& strategyFile,
           bool _enable_control_replication,
           bool _log_instance_creation);
  ~FFMapper();
  virtual const char* get_mapper_name(void) const;
  virtual MapperSyncModel get_mapper_sync_model(void) const;
public:
  static void register_sharding_functor(int argv, char** argc);
  virtual void select_task_options(const MapperContext    ctx,
                                   const Task&            task,
                                         TaskOptions&     output);
  virtual void premap_task(const MapperContext      ctx,
                           const Task&              task, 
                           const PremapTaskInput&   input,
                           PremapTaskOutput&        output);
  virtual void slice_task(const MapperContext      ctx,
                          const Task&              task, 
                          const SliceTaskInput&    input,
                                SliceTaskOutput&   output);
  virtual void map_task(const MapperContext      ctx,
                        const Task&              task,
                        const MapTaskInput&      input,
                              MapTaskOutput&     output);
  virtual void map_replicate_task(const MapperContext      ctx,
                                  const Task&              task,
                                  const MapTaskInput&      input,
                                  const MapTaskOutput&     default_output,
                                  MapReplicateTaskOutput&  output);
  virtual void select_task_variant(const MapperContext          ctx,
                                   const Task&                  task,
                                   const SelectVariantInput&    input,
                                         SelectVariantOutput&   output);
  virtual void postmap_task(const MapperContext      ctx,
                            const Task&              task,
                            const PostMapInput&      input,
                                  PostMapOutput&     output);
  virtual void select_task_sources(const MapperContext        ctx,
                                   const Task&                task,
                                   const SelectTaskSrcInput&  input,
                                         SelectTaskSrcOutput& output);
  virtual void create_task_temporary_instance(
                                const MapperContext              ctx,
                                const Task&                      task,
                                const CreateTaskTemporaryInput&  input,
                                      CreateTaskTemporaryOutput& output);
  virtual void speculate(const MapperContext      ctx,
                         const Task&              task,
                               SpeculativeOutput& output);
  virtual void report_profiling(const MapperContext      ctx,
                                const Task&              task,
                                const TaskProfilingInfo& input);
  virtual void select_sharding_functor(
                             const MapperContext                ctx,
                             const Task&                        task,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output);
public: // Inline mapping calls
  virtual void map_inline(const MapperContext        ctx,
                          const InlineMapping&       inline_op,
                          const MapInlineInput&      input,
                                MapInlineOutput&     output);
  virtual void select_inline_sources(const MapperContext        ctx,
                                   const InlineMapping&         inline_op,
                                   const SelectInlineSrcInput&  input,
                                         SelectInlineSrcOutput& output);
  virtual void create_inline_temporary_instance(
                              const MapperContext                ctx,
                              const InlineMapping&               inline_op,
                              const CreateInlineTemporaryInput&  input,
                                    CreateInlineTemporaryOutput& output);
  virtual void report_profiling(const MapperContext         ctx,
                                const InlineMapping&        inline_op,
                                const InlineProfilingInfo&  input);
public: // Copy mapping calls
  virtual void map_copy(const MapperContext      ctx,
                        const Copy&              copy,
                        const MapCopyInput&      input,
                              MapCopyOutput&     output);
  virtual void select_copy_sources(const MapperContext          ctx,
                                   const Copy&                  copy,
                                   const SelectCopySrcInput&    input,
                                         SelectCopySrcOutput&   output);
  virtual void create_copy_temporary_instance(
                              const MapperContext              ctx,
                              const Copy&                      copy,
                              const CreateCopyTemporaryInput&  input,
                                    CreateCopyTemporaryOutput& output);
  virtual void speculate(const MapperContext      ctx,
                         const Copy& copy,
                               SpeculativeOutput& output);
  virtual void report_profiling(const MapperContext      ctx,
                                const Copy&              copy,
                                const CopyProfilingInfo& input);
  virtual void select_sharding_functor(
                             const MapperContext                ctx,
                             const Copy&                        copy,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output);
public: // Close mapping calls
  virtual void map_close(const MapperContext       ctx,
                         const Close&              close,
                         const MapCloseInput&      input,
                               MapCloseOutput&     output);
  virtual void select_close_sources(const MapperContext        ctx,
                                    const Close&               close,
                                    const SelectCloseSrcInput&  input,
                                          SelectCloseSrcOutput& output);
  virtual void create_close_temporary_instance(
                              const MapperContext               ctx,
                              const Close&                      close,
                              const CreateCloseTemporaryInput&  input,
                                    CreateCloseTemporaryOutput& output);
  virtual void report_profiling(const MapperContext       ctx,
                                const Close&              close,
                                const CloseProfilingInfo& input);
  virtual void select_sharding_functor(
                             const MapperContext                ctx,
                             const Close&                       close,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output);
public: // Acquire mapping calls
  virtual void map_acquire(const MapperContext         ctx,
                           const Acquire&              acquire,
                           const MapAcquireInput&      input,
                                 MapAcquireOutput&     output);
  virtual void speculate(const MapperContext         ctx,
                         const Acquire&              acquire,
                               SpeculativeOutput&    output);
  virtual void report_profiling(const MapperContext         ctx,
                                const Acquire&              acquire,
                                const AcquireProfilingInfo& input);
  virtual void select_sharding_functor(
                             const MapperContext                ctx,
                             const Acquire&                     acquire,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output);
public: // Release mapping calls
  virtual void map_release(const MapperContext         ctx,
                           const Release&              release,
                           const MapReleaseInput&      input,
                                 MapReleaseOutput&     output);
  virtual void select_release_sources(const MapperContext       ctx,
                                 const Release&                 release,
                                 const SelectReleaseSrcInput&   input,
                                       SelectReleaseSrcOutput&  output);
  virtual void create_release_temporary_instance(
                               const MapperContext                 ctx,
                               const Release&                      release,
                               const CreateReleaseTemporaryInput&  input,
                                     CreateReleaseTemporaryOutput& output);
  virtual void speculate(const MapperContext         ctx,
                         const Release&              release,
                               SpeculativeOutput&    output);
  virtual void report_profiling(const MapperContext         ctx,
                                const Release&              release,
                                const ReleaseProfilingInfo& input);
  virtual void select_sharding_functor(
                             const MapperContext                ctx,
                             const Release&                     release,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output);
public: // Partition mapping calls
  virtual void select_partition_projection(const MapperContext  ctx,
                      const Partition&                          partition,
                      const SelectPartitionProjectionInput&     input,
                            SelectPartitionProjectionOutput&    output);
  virtual void map_partition(const MapperContext        ctx,
                             const Partition&           partition,
                             const MapPartitionInput&   input,
                                   MapPartitionOutput&  output);
  virtual void select_partition_sources(
                               const MapperContext             ctx,
                               const Partition&                partition,
                               const SelectPartitionSrcInput&  input,
                                     SelectPartitionSrcOutput& output);
  virtual void create_partition_temporary_instance(
                          const MapperContext                   ctx,
                          const Partition&                      partition,
                          const CreatePartitionTemporaryInput&  input,
                                CreatePartitionTemporaryOutput& output);
  virtual void report_profiling(const MapperContext              ctx,
                                const Partition&                 partition,
                                const PartitionProfilingInfo&    input);
  virtual void select_sharding_functor(
                             const MapperContext                ctx,
                             const Partition&                   partition,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output);
public: // Fill mapper calls
  virtual void select_sharding_functor(
                             const MapperContext                ctx,
                             const Fill&                        fill,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output);
public: // Task execution mapping calls
  virtual void configure_context(const MapperContext         ctx,
                                 const Task&                 task,
                                       ContextConfigOutput&  output);
  virtual void select_tunable_value(const MapperContext         ctx,
                                    const Task&                 task,
                                    const SelectTunableInput&   input,
                                          SelectTunableOutput&  output);
public: // Must epoch mapping
  virtual void select_sharding_functor(
                      const MapperContext                    ctx,
                      const MustEpoch&                       epoch,
                      const SelectShardingFunctorInput&      input,
                            MustEpochShardingFunctorOutput&  output);
  virtual void map_must_epoch(const MapperContext           ctx,
                              const MapMustEpochInput&      input,
                                    MapMustEpochOutput&     output);
public: // Dataflow graph mapping
  virtual void map_dataflow_graph(const MapperContext           ctx,
                                  const MapDataflowGraphInput&  input,
                                        MapDataflowGraphOutput& output);
public: // Memoization control
  virtual void memoize_operation(const MapperContext  ctx,
                                 const Mappable&      mappable,
                                 const MemoizeInput&  input,
                                       MemoizeOutput& output);
public: // Mapping control and stealing
  virtual void select_tasks_to_map(const MapperContext          ctx,
                                   const SelectMappingInput&    input,
                                         SelectMappingOutput&   output);
  virtual void select_steal_targets(const MapperContext         ctx,
                                    const SelectStealingInput&  input,
                                          SelectStealingOutput& output);
  virtual void permit_steal_request(const MapperContext         ctx,
                                    const StealRequestInput&    intput,
                                          StealRequestOutput&   output);
private: //static inline methods
  static inline bool physical_sort_func(
                     const std::pair<PhysicalInstance,unsigned> &left,
                     const std::pair<PhysicalInstance,unsigned> &right)
    { return (left.second < right.second); }
private: // Default helper functions
  Memory default_select_target_memory(MapperContext ctx,
		                      Processor target_proc,
				      const RegionRequirement &req);
  bool default_make_instance(MapperContext ctx,
                             Memory target_mem,
                             const LayoutConstraintSet &constraints,
                             PhysicalInstance &result,
                             bool meets_constraints,
                             const RegionRequirement &req,
                             bool &created,
                             size_t *footprint);
  LayoutConstraintID  default_select_layout_constraints(MapperContext ctx,
                                         Memory target_memory,
                                         const RegionRequirement &req,
                                         bool needs_field_constraint_check);
  void default_select_constraints(MapperContext ctx,
                                  LayoutConstraintSet &constraints,
                                  Memory target_memory,
                                  const RegionRequirement &req);
  void default_policy_select_sources(MapperContext ctx,
                                     const PhysicalInstance &target,
                                     const std::vector<PhysicalInstance> &sources,
                                     std::deque<PhysicalInstance> &ranking,
                                     Memory preferred_memory = Memory::NO_MEMORY);
private:
  unsigned long long compute_task_hash(const Task& task);
  bool is_parameter_server_update_task(TaskID tid);
  bool is_initializer_task(TaskID tid);
  const std::vector<Processor>& all_procs_by_kind(Processor::Kind kind);
protected:
  const Processor local_processor;
  const AddressSpace node_id;
  size_t total_nodes;
  const char* mapper_name;
  bool enable_control_replication;
  bool log_instance_creation;
  std::vector<Processor> all_gpus, all_cpus, all_pys, local_gpus, local_cpus, local_pys;
  std::map<Processor, Memory> proc_fbmems, proc_zcmems;
  std::map<unsigned long long, Processor> cache_update_tasks;
  // We use MappingTagID has the key since we will pass the tag to the mapper
  std::map<MappingTagID, ParallelConfig> strategies;
  std::map<std::pair<Memory::Kind,FieldSpace>, LayoutConstraintID> layout_constraint_cache;
  std::vector<InstanceCreationLog> created_instances;
};

#ifdef DEADCODE
class FFMapper : public DefaultMapper {
public:
  FFMapper(MapperRuntime *rt, Machine machine, Processor local,
           const char *mapper_name,
           std::map<MappingTagID, ParallelConfig>* strategies);
public:
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       const Task& task,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output);
  virtual void select_task_options(const MapperContext ctx,
                                   const Task& task,
                                   TaskOptions& output);
  virtual Memory default_policy_select_target_memory(MapperContext ctx,
                                                     Processor target_proc,
                                                     const RegionRequirement &req,
                                                     MemoryConstraint mc);
  virtual void map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                        MapTaskOutput& output);
protected:
  std::vector<Processor>& gpus;
  std::map<Processor, Memory>& proc_fbmems, proc_zcmems;
  std::vector<Processor>& cpus;
  std::map<unsigned long long, Processor> cache_update_tasks;
  // We use MappingTagID has the key since we will pass the tag to the mapper
  std::map<MappingTagID, ParallelConfig>& strategies;
};
#endif

void update_mappers(Machine machine, Runtime *rt, const std::set<Processor> &local_procs);
#endif
