/* Copyright 2020 Facebook, Stanford University
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

#include "mapper.h"

LegionRuntime::Logger::Category log_ff_mapper("Mapper");

FFMapper::FFMapper(MapperRuntime *rt, Machine machine, Processor local,
                   const char *_mapper_name,
                   const std::string& strategyFile,
                   bool _enable_control_replication)
  : NullMapper(rt, machine), node_id(local.address_space()),
    mapper_name(_mapper_name),
    enable_control_replication(_enable_control_replication)
{
  std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);
  Machine::ProcessorQuery proc_query(machine);
  std::set<AddressSpace> address_space_set;
  for (Machine::ProcessorQuery::iterator it = proc_query.begin();
      it != proc_query.end(); it++)
  {
    address_space_set.insert(it->address_space());
    if (it->kind() == Processor::TOC_PROC) {
      all_gpus.push_back(*it);
      if (it->address_space() == node_id)
        local_gpus.push_back(*it);
      Machine::MemoryQuery fb_query(machine);
      fb_query.only_kind(Memory::GPU_FB_MEM);
      fb_query.best_affinity_to(*it);
      assert(fb_query.count() == 1);
      proc_fbmems[*it] = *(fb_query.begin());
      Machine::MemoryQuery zc_query(machine);
      zc_query.only_kind(Memory::Z_COPY_MEM);
      zc_query.has_affinity_to(*it);
      assert(zc_query.count() == 1);
      proc_zcmems[*it] = *(zc_query.begin());
    } else if (it->kind() == Processor::LOC_PROC) {
      all_cpus.push_back(*it);
      if (it->address_space() == node_id)
        local_cpus.push_back(*it);
      Machine::MemoryQuery zc_query(machine);
      zc_query.only_kind(Memory::Z_COPY_MEM);
      zc_query.has_affinity_to(*it);
      assert(zc_query.count() == 1);
      proc_zcmems[*it] = *(zc_query.begin());
    } else if (it->kind() == Processor::PY_PROC) {
      all_pys.push_back(*it);
      if (it->address_space() == node_id)
        local_pys.push_back(*it);
      Machine::MemoryQuery zc_query(machine);
      zc_query.only_kind(Memory::Z_COPY_MEM);
      zc_query.has_affinity_to(*it);
      assert(zc_query.count() == 1);
      proc_zcmems[*it] = *(zc_query.begin());
    }
  }
  total_nodes = address_space_set.size();
  if (enable_control_replication)
    log_ff_mapper.print("Enabled Control Replication Optimizations.");
  if (strategyFile == "") {
    // No strategy file provided, use data parallelism
    log_ff_mapper.print("No strategy file provided. Use default data parallelism.");
  } else {
    log_ff_mapper.print("Load parallelization strategy from file %s",
                     strategyFile.c_str());
    load_strategies_from_file(strategyFile, strategies);
  }
  int start_dim = 1, end_dim = 4;
#if MAX_TENSOR_DIM >= 5
  end_dim = 5;
#endif
  for (int i = start_dim; i <= end_dim; i++) {
    ParallelConfig pc;
    pc.device_type = ParallelConfig::GPU;
    pc.nDims = i;
    for (int j = 0; j < pc.nDims; j++)
      pc.dim[j] = 1;
    pc.dim[pc.nDims-1] = all_gpus.size();
    for (size_t j = 0; j < all_gpus.size(); j++)
      pc.device_ids[j] = j;
    strategies[FFConfig::DataParallelism_GPU_1D+i-1] = pc;
  }
  for (int i = start_dim; i <= end_dim; i++) {
    ParallelConfig pc;
    pc.device_type = ParallelConfig::CPU;
    pc.nDims = i;
    for (int j = 0; j < pc.nDims; j++)
      pc.dim[j] = 1;
    pc.dim[pc.nDims-1] = all_cpus.size();
    for (size_t j = 0; j < all_cpus.size(); j++)
      pc.device_ids[j] = j;
    strategies[FFConfig::DataParallelism_CPU_1D+i-1] = pc;
  }
}

bool FFMapper::is_parameter_server_update_task(TaskID tid)
{
  switch (tid) {
    case SGD_UPD_PS_TASK_ID:
    case ADAM_UPD_PS_TASK_ID:
      return true;
    default:
      return false;
  }
}

bool FFMapper::is_initializer_task(TaskID tid)
{
  switch (tid) {
    case GLOROT_INIT_TASK_ID:
    case ZERO_INIT_TASK_ID:
    case CONSTANT_INIT_TASK_ID:
    case UNIFORM_INIT_TASK_ID:
    case NORMAL_INIT_TASK_ID:
      return true;
    default:
      return false;
  }
}

const char* FFMapper::get_mapper_name(void) const
{
  return mapper_name;
}

Mapper::MapperSyncModel FFMapper::get_mapper_sync_model(void) const
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

void FFMapper::select_task_options(const MapperContext ctx,
                                   const Task& task,
                                   TaskOptions& output)
{
  unsigned long long task_hash = compute_task_hash(task);
  output.inline_task = false;
  output.stealable = false;
  output.map_locally = true;

  if (task.task_id == STRATEGY_SEARCH_TASK_ID) {
    output.initial_proc = local_gpus[0];
    return;
  }
  if (task.task_id == UPDATE_METRICS_TASK_ID) {
    output.initial_proc = local_cpus[0];
    return;
  }
  if (task.task_id == TOP_LEVEL_TASK_ID) {
    output.initial_proc = local_cpus[0];
    // control replicate top level task
    if (enable_control_replication) {
      output.replicate = true;
    }
    return;
  }
  if (task.task_id == PYTHON_TOP_LEVEL_TASK_ID) {
    output.initial_proc = local_pys[0];
    // control replicate python top level task
    if (enable_control_replication) {
      output.replicate = true;
    }
    return;
  }

  if (is_parameter_server_update_task(task.task_id)
  || is_initializer_task(task.task_id)) {
    // For Parameter Server Update, pick a processor from config
    MappingTagID hash = task.tag;
    ParallelConfig config;
    if (strategies.find(hash) != strategies.end()) {
      config = strategies[hash];
      int num_parts = 1;
      for (int i = 0; i < config.nDims; i++)
        num_parts *= config.dim[i];
      if (num_parts == 1) {
        output.initial_proc = all_gpus[config.device_ids[0]];
        // Current assert this sould be a local proc
        assert(output.initial_proc.address_space() == node_id);
        return;
      } else {
        output.initial_proc = all_gpus[config.device_ids[0]];
        return;
      }
    }
    if (cache_update_tasks.find(task_hash) != cache_update_tasks.end()) {
      output.initial_proc = cache_update_tasks[task_hash];
      assert(output.initial_proc.address_space() == node_id);
      return;
    }
    // randomly select a local processor
    output.initial_proc = local_gpus[task_hash % local_gpus.size()];
    cache_update_tasks[task_hash] = output.initial_proc;
    return;
  }

  if ((task.task_id >= CUSTOM_CPU_TASK_ID_FIRST)
     && (task.task_id <= CUSTOM_CPU_TASK_ID_LAST))
  {
    if (!task.is_index_space) {
      output.initial_proc = local_cpus[0];
      return;
    }
  }

  if ((task.task_id == PY_DL_FLOAT_LOAD_ENTIRE_CPU_TASK_ID)
    || (task.task_id == PY_DL_INT_LOAD_ENTIRE_CPU_TASK_ID)
    || (task.task_id == PY_DL_FLOAT_INDEX_LOAD_ENTIRE_CPU_TASK_ID)
    || (task.task_id == PY_DL_INT_INDEX_LOAD_ENTIRE_CPU_TASK_ID))
  {
    if (!task.is_index_space) {
      output.initial_proc = local_cpus[0];
      return;
    }
  }

  // Assert that all single tasks should be handled and returned before
  // So task must be an indextask
  if (!task.is_index_space) {
    fprintf(stderr, "The following task is currently not captured by the "
            "FlexFlow Mapper: %s\n"
            "Report the issue to the FlexFlow developers",
            task.get_task_name());
  }
  assert(task.is_index_space);
}

void FFMapper::slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output)
{
  output.slices.resize(input.domain.get_volume());
  const std::vector<Processor>* devices;
  ParallelConfig config;
  if ((task.task_id == TOP_LEVEL_TASK_ID)
  || ((task.task_id >= CUSTOM_CPU_TASK_ID_FIRST)
     && (task.task_id <= CUSTOM_CPU_TASK_ID_LAST))) {
    int ndim = input.domain.get_dim();
    assert(strategies.find(FFConfig::DataParallelism_CPU_1D-1+ndim) != strategies.end());
    config = strategies[FFConfig::DataParallelism_CPU_1D-1+ndim];
    printf("num_parts %d", config.num_parts());
    devices = &all_cpus;
  } else if ((task.task_id == PY_DL_FLOAT_INDEX_LOAD_ENTIRE_CPU_TASK_ID)
  || (task.task_id == PY_DL_INT_INDEX_LOAD_ENTIRE_CPU_TASK_ID)) {
    // FIXME: even though it is a CPU task, we use data parallelism
    assert(enable_control_replication);
    int ndim = input.domain.get_dim();
    assert(strategies.find(FFConfig::DataParallelism_GPU_1D-1+ndim) != strategies.end());
    config = strategies[FFConfig::DataParallelism_GPU_1D-1+ndim];
    devices = &all_cpus;
  } else {
    MappingTagID hash = task.tag;
    // Make sure the task has a non-zero tag
    assert(hash != 0);
    if (strategies.find(hash) == strategies.end()) {
      // No strategy found, use default data parallelism
      int ndim = input.domain.get_dim();
      assert(strategies.find(FFConfig::DataParallelism_GPU_1D-1+ndim) != strategies.end());
      config = strategies[FFConfig::DataParallelism_GPU_1D-1+ndim];
    } else {
      // Found a strategy
      config = strategies[hash];
      // Check that the dimensions match
      assert(config.nDims == input.domain.get_dim());
    }
    if (config.device_type == ParallelConfig::GPU) {
      devices = &all_gpus;
    } else {
      devices = &all_cpus;
    }
  }
  switch (input.domain.get_dim())
  {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = input.domain; \
      int cnt = 0; \
      for (PointInRectIterator<DIM> pir(rect); pir(); pir++) { \
        int idx = 0; \
        for (int i = input.domain.get_dim()-1; i >= 0; i--) \
          idx = idx*(task.index_domain.hi()[i]-task.index_domain.lo()[i]+1)+pir[i]-task.index_domain.lo()[i]; \
        assert(config.num_parts() > idx); \
        Rect<DIM> slice(*pir, *pir); \
        output.slices[cnt++] = TaskSlice(slice, \
            (*devices)[config.device_ids[idx] % devices->size()], \
            false/*recurse*/, false/*stealable*/); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  // In control replication, each mapper should only receive task slices
  // that should be assigned to local proccessors
  // Violation of this assertion may result in severe runtime overheads
  // to Legion
  if (enable_control_replication) {
    for (size_t i = 0; i < output.slices.size(); i++) {
      assert(output.slices[i].proc.address_space() == node_id);
    }
  }
}

void FFMapper::premap_task(const MapperContext      ctx,
                           const Task&              task,
                           const PremapTaskInput&   input,
                           PremapTaskOutput&        output)
{
  assert(false);
}

void FFMapper::map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                        MapTaskOutput& output)
{
  std::vector<VariantID> variant_ids;
  runtime->find_valid_variants(ctx, task.task_id, variant_ids, task.target_proc.kind());
  // Currently assume there is exactly one variant
  assert(variant_ids.size() == 1);
  output.chosen_variant = variant_ids[0];
  // TODO: assign priorities
  output.task_priority = 0;
  output.postmap_task = false;
  if (task.target_proc.address_space() != node_id) {
    output.target_procs.push_back(task.target_proc);
  } else if (task.target_proc.kind() == Processor::TOC_PROC) {
    output.target_procs.push_back(task.target_proc);
  } else if (task.target_proc.kind() == Processor::LOC_PROC) {
    // Put any of our CPU procs here
    // If we're part of a must epoch launch, our
    // target proc will be sufficient
    if (!task.must_epoch_task)
      output.target_procs.insert(output.target_procs.end(),
          local_cpus.begin(), local_cpus.end());
    else
      output.target_procs.push_back(task.target_proc);
  } else if (task.target_proc.kind() == Processor::PY_PROC) {
    // Put any of our Python procs here
    // If we're part of a must epoch launch, our
    // target proc will be sufficient
    if (!task.must_epoch_task)
      output.target_procs.insert(output.target_procs.end(),
          local_pys.begin(), local_pys.end());
    else
      output.target_procs.push_back(task.target_proc);
  } else {
    // Unsupported proc kind
    assert(false);
  }
  // In control replication, each mapper should only map tasks
  // assigned to local proccessors
  // Violation of this assertion may result in severe runtime
  // overheads to Legion
  if (enable_control_replication) {
    for (size_t i = 0; i < output.target_procs.size(); i++)
      assert(output.target_procs[i].address_space() == node_id);
  }
  // Find instances that still need to be mapped
  std::vector<std::set<FieldID> > missing_fields(task.regions.size());
  runtime->filter_instances(ctx, task, output.chosen_variant,
      output.chosen_instances, missing_fields);
  // Track which regions have already been mapped
  std::vector<bool> done_regions(task.regions.size(), false);
  if (!input.premapped_regions.empty())
    for (std::vector<unsigned>::const_iterator it =
          input.premapped_regions.begin(); it !=
          input.premapped_regions.end(); it++)
      done_regions[*it] = true;
  // Now we need to go through and make instances for any of our
  // regions which do not have space for certain fields
  for (unsigned idx = 0; idx < task.regions.size(); idx++) {
    if (done_regions[idx])
      continue;
    // Skip any empty regions
    if ((task.regions[idx].privilege == LEGION_NO_ACCESS) ||
        (task.regions[idx].privilege_fields.empty()) ||
        missing_fields[idx].empty())
      continue;
    // Select a memory for the req
    Memory target_mem = default_select_target_memory(ctx,
        task.target_proc, task.regions[idx]);
    // Assert no virtual mapping for now
    assert((task.regions[idx].tag & DefaultMapper::VIRTUAL_MAP) == 0);
    // Check to see if any of the valid instances satisfy the requirement
    {
      std::vector<PhysicalInstance> valid_instances;
      for (std::vector<PhysicalInstance>::const_iterator
             it = input.valid_instances[idx].begin(),
             ie = input.valid_instances[idx].end(); it != ie; ++it)
      {
        if (it->get_location() == target_mem) {
          // Only select instances with exact same index domain
          Domain instance_domain = it->get_instance_domain();
          Domain region_domain = runtime->get_index_space_domain(
              ctx, task.regions[idx].region.get_index_space());
          if (instance_domain.get_volume() == region_domain.get_volume()) 
            valid_instances.push_back(*it);
        }
      }

      std::set<FieldID> valid_missing_fields;
      runtime->filter_instances(ctx, task, idx, output.chosen_variant,
                                valid_instances, valid_missing_fields);
      runtime->acquire_and_filter_instances(ctx, valid_instances);
      output.chosen_instances[idx] = valid_instances;
      missing_fields[idx] = valid_missing_fields;
      if (missing_fields[idx].empty())
        continue;
    }
    // Otherwise make nromal instances for the given region
    LayoutConstraintID layout_id = default_select_layout_constraints(
        ctx, target_mem, task.regions[idx], false/*needs constraint check*/);
    const LayoutConstraintSet &constraint_set =
        runtime->find_layout_constraints(ctx, layout_id);
    size_t footprint;
    PhysicalInstance result;
    if (!default_make_instance(ctx, target_mem, constraint_set,
        result, true/*meet_constraints*/,
        task.regions[idx], &footprint))
    {
      // Report failed to creation
      log_ff_mapper.error("FlexFlow failed allocation of size %zd bytes for "
          "region requirement %d of task %s (UID %lld) in memory "
          IDFMT " with kind %d for processor " IDFMT ".", footprint, idx,
          task.get_task_name(), task.get_unique_id(),
          target_mem.id, target_mem.kind(), task.target_proc.id);
      assert(false);
    } else {
      output.chosen_instances[idx].push_back(result);
    }
  } //for idx
#ifdef DEADCODE
  if ((task.task_id == CONV2D_INIT_TASK_ID)
     || (task.task_id == CONV2D_FWD_TASK_ID)
     || (task.task_id == CONV2D_BWD_TASK_ID))
  {
    VariantInfo chosen = default_find_preferred_variant(task, ctx,
                             true/*needs tight bound*/, true/*cache*/,
                             task.target_proc.kind());
    output.chosen_variant = chosen.variant;
    output.task_priority = 0;
    output.postmap_task = false;
    output.target_procs.push_back(task.target_proc);
    assert(task.target_proc.kind() == Processor::TOC_PROC);
    Memory fbmem = proc_fbmems[task.target_proc];
    for (unsigned idx = 0; idx < task.regions.size(); idx++)
    {
      if ((task.regions[idx].privilege == NO_ACCESS) ||
          (task.regions[idx].privilege_fields.empty())) continue;
      const TaskLayoutConstraintSet &layout_constraints =
        runtime->find_task_layout_constraints(ctx, task.task_id,
                                              output.chosen_variant);
      std::set<FieldID> fields(task.regions[idx].privilege_fields);
      if (!default_create_custom_instances(ctx, task.target_proc,
             fbmem, task.regions[idx], idx, fields,
             layout_constraints, true, output.chosen_instances[idx]))
      {
        default_report_failed_instance_creation(task, idx, task.target_proc,
                                                fbmem);
      }
    }
  } else
#endif
}

void FFMapper::map_replicate_task(const MapperContext      ctx,
                                  const Task&              task,
                                  const MapTaskInput&      input,
                                  const MapTaskOutput&     default_output,
                                  MapReplicateTaskOutput&  output)
{
  // Should only be replicated for the top-level task
  assert((task.get_depth() == 0) && (task.regions.size() == 0));
  const Processor::Kind target_kind = task.target_proc.kind();
  VariantID chosen_variant;
  {
    std::vector<VariantID> variant_ids;
    runtime->find_valid_variants(ctx, task.task_id, variant_ids, task.target_proc.kind());
    // Currently assume there is exactly one variant
    assert(variant_ids.size() == 1);
    chosen_variant = variant_ids[0];
  }
  const std::vector<Processor> &all_procs = all_procs_by_kind(target_kind);
  // Place on replicate on each node by default
  output.task_mappings.resize(total_nodes, default_output);
  // Assume default_output does not include any target_procs
  assert(default_output.target_procs.size() == 0);
  for (std::vector<Processor>::const_iterator it = all_procs.begin();
      it != all_procs.end(); it++)
  {
    AddressSpace space = it->address_space();
    assert(space < output.task_mappings.size());
    // Add *it as a target_proc if we haven't found one
    if (output.task_mappings[space].target_procs.size() == 0) {
      output.task_mappings[space].target_procs.push_back(*it);
    }
  }
  output.control_replication_map.resize(total_nodes);
  for (unsigned idx = 0; idx < total_nodes; idx++)
  {
    output.task_mappings[idx].chosen_variant = chosen_variant;
    output.control_replication_map[idx] =
        output.task_mappings[idx].target_procs[0];
  }
}

void FFMapper::select_task_variant(const MapperContext          ctx,
                                   const Task&                  task,
                                   const SelectVariantInput&    input,
                                         SelectVariantOutput&   output)
{
  assert(false);
}

void FFMapper::postmap_task(const MapperContext      ctx,
                            const Task&              task,
                            const PostMapInput&      input,
                                  PostMapOutput&     output)
{
  assert(false);
}

void FFMapper::select_task_sources(const MapperContext        ctx,
                                   const Task&                task,
                                   const SelectTaskSrcInput&  input,
                                         SelectTaskSrcOutput& output)
{
  if (task.task_id == PS_PREFETCH_TASK_ID) {
    // Dummy task refers to prefetching weights tasks
    MappingTagID hash = task.tag;
    assert(hash != 0);
    ParallelConfig config;
    if (strategies.find(hash) == strategies.end()) {
      // No strategy found, use default data parallelism
      assert(strategies.find(FFConfig::DataParallelism_GPU_2D) != strategies.end());
      config = strategies[FFConfig::DataParallelism_GPU_2D];
    } else {
      // Found a strategy
      config = strategies[hash];
    }
    Processor parameter_server = all_gpus[config.device_ids[0]];
    // Prefer instances located on the parameter server
    Memory ps_memory = proc_fbmems[parameter_server];
    default_policy_select_sources(ctx, input.target, input.source_instances,
        output.chosen_ranking, ps_memory);
    return;
  }
  default_policy_select_sources(ctx, input.target, input.source_instances,
      output.chosen_ranking);
}

void FFMapper::default_policy_select_sources(MapperContext ctx,
                                             const PhysicalInstance &target,
                                             const std::vector<PhysicalInstance> &sources,
                                             std::deque<PhysicalInstance> &ranking,
                                             Memory preferred_memory)
{
  // We rank source instances by the bandwidth of the memory
  // they are in to the destination
  std::map<Memory, unsigned> source_memories;
  Memory destination_memory = target.get_location();
  std::vector<MemoryMemoryAffinity> affinity(1);
  std::vector<std::pair<PhysicalInstance, unsigned> > band_ranking(sources.size());
  for (unsigned idx = 0; idx < sources.size(); idx++) {
    const PhysicalInstance &instance = sources[idx];
    Memory location = instance.get_location();
    std::map<Memory, unsigned>::const_iterator finder =
        source_memories.find(location);
    if (finder == source_memories.end()) {
      affinity.clear();
      machine.get_mem_mem_affinity(affinity, location, destination_memory, false);
      unsigned memory_bandwidth = 0;
      if (!affinity.empty()) {
        assert(affinity.size() == 1);
        memory_bandwidth = affinity[0].bandwidth;
        // Add 1000 points to the bandwidth to prioritize preferred_memory
        if (preferred_memory == location)
          memory_bandwidth += 1000;
      }
      source_memories[location] = memory_bandwidth;
      band_ranking[idx] =
          std::pair<PhysicalInstance, unsigned>(instance, memory_bandwidth);
    } else {
      band_ranking[idx] =
          std::pair<PhysicalInstance, unsigned>(instance, finder->second);
    }
  }
  // Sort them by bandwidth
  std::sort(band_ranking.begin(), band_ranking.end(), physical_sort_func);
  // Iterate from largest bandwidth to smallest
  for (std::vector<std::pair<PhysicalInstance,unsigned> >::
        const_reverse_iterator it = band_ranking.rbegin();
        it != band_ranking.rend(); it++) {
    ranking.push_back(it->first);
    break;
  }
}

void FFMapper::create_task_temporary_instance(
                                const MapperContext              ctx,
                                const Task&                      task,
                                const CreateTaskTemporaryInput&  input,
                                      CreateTaskTemporaryOutput& output)
{
  assert(false);
}

void FFMapper::speculate(const MapperContext      ctx,
                         const Task&              task,
                               SpeculativeOutput& output)
{
  assert(false);
}

void FFMapper::report_profiling(const MapperContext      ctx,
                                const Task&              task,
                                const TaskProfilingInfo& input)
{
  assert(false);
}

void FFMapper::select_sharding_functor(const MapperContext ctx,
                                       const Task& task,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output)
{
  // Current all shardings uses data parallelism across machines
  output.chosen_functor = DataParallelShardingID;
}

void FFMapper::map_inline(const MapperContext        ctx,
                          const InlineMapping&       inline_op,
                          const MapInlineInput&      input,
                                MapInlineOutput&     output)
{
  LayoutConstraintSet creation_constraints;
  Memory target_memory = Memory::NO_MEMORY;
  if (inline_op.layout_constraint_id > 0) {
    // Find our constraints
    creation_constraints = runtime->find_layout_constraints(ctx,
                                        inline_op.layout_constraint_id);
    if (creation_constraints.memory_constraint.is_valid())
    {
      Machine::MemoryQuery valid_mems(machine);
      valid_mems.has_affinity_to(inline_op.parent_task->current_proc);
      valid_mems.only_kind(
          creation_constraints.memory_constraint.get_kind());
      if (valid_mems.count() == 0)
      {
        log_ff_mapper.error("FlexFlow mapper error. Mapper %s could find no "
            "valid memories for the constraints requested by "
            "inline mapping %lld in parent task %s (ID %lld).",
            get_mapper_name(), inline_op.get_unique_id(),
            inline_op.parent_task->get_task_name(),
            inline_op.parent_task->get_unique_id());
        assert(false);
      }
      target_memory = valid_mems.first(); // just take the first one
    }
    else
      target_memory = default_select_target_memory(ctx,
          inline_op.parent_task->current_proc, inline_op.requirement);
    if (creation_constraints.field_constraint.field_set.empty())
      creation_constraints.add_constraint(FieldConstraint(
            inline_op.requirement.privilege_fields, false/*contig*/));
  } else {
    // No constraints so do what we want
    target_memory = default_select_target_memory(ctx,
        inline_op.parent_task->current_proc, inline_op.requirement);
    // Copy over any valid instances for our target memory, then try to
    // do an acquire on them and see which instances are no longer valid
    if (!input.valid_instances.empty())
    {
      for (std::vector<PhysicalInstance>::const_iterator it =
            input.valid_instances.begin(); it !=
            input.valid_instances.end(); it++)
      {
        if (it->get_location() == target_memory)
          output.chosen_instances.push_back(*it);
      }
      if (!output.chosen_instances.empty())
        runtime->acquire_and_filter_instances(ctx,
                                          output.chosen_instances);
    }
    // Now see if we have any fields which we still make space for
    std::set<FieldID> missing_fields =
      inline_op.requirement.privilege_fields;
    for (std::vector<PhysicalInstance>::const_iterator it =
          output.chosen_instances.begin(); it !=
          output.chosen_instances.end(); it++)
    {
      it->remove_space_fields(missing_fields);
      if (missing_fields.empty())
        break;
    }
    // If we've satisfied all our fields, then we are done
    if (missing_fields.empty())
      return;
    // Otherwise, let's make an instance for our missing fields
    LayoutConstraintID our_layout_id = default_select_layout_constraints(
        ctx, target_memory, inline_op.requirement, true);
    creation_constraints = runtime->find_layout_constraints(ctx, our_layout_id);
    creation_constraints.add_constraint(
        FieldConstraint(missing_fields, false/*contig*/, false/*inorder*/));
  }
  PhysicalInstance result;
  size_t footprint;
  if (!default_make_instance(ctx, target_memory, creation_constraints,
      result, true/*meets_constraints*/, inline_op.requirement, &footprint))
  {
    log_ff_mapper.error("FlexFlow Mapper failed allocation of size %zd bytes"
        " for region requirement of inline ammping in task %s (UID %lld)"
        " in memory " IDFMT "for processor " IDFMT ".", footprint,
        inline_op.parent_task->get_task_name(),
        inline_op.parent_task->get_unique_id(),
        target_memory.id,
        inline_op.parent_task->current_proc.id);
    assert(false);
  } else {
    output.chosen_instances.push_back(result);
  }
}

void FFMapper::select_inline_sources(const MapperContext        ctx,
                                     const InlineMapping&         inline_op,
                                     const SelectInlineSrcInput&  input,
                                           SelectInlineSrcOutput& output)
{
  //assert(false);
  default_policy_select_sources(ctx, input.target, input.source_instances,
                                output.chosen_ranking);
}

void FFMapper::create_inline_temporary_instance(
                              const MapperContext                ctx,
                              const InlineMapping&               inline_op,
                              const CreateInlineTemporaryInput&  input,
                                    CreateInlineTemporaryOutput& output)
{
  assert(false);
}

void FFMapper::report_profiling(const MapperContext         ctx,
                                const InlineMapping&        inline_op,
                                const InlineProfilingInfo&  input)
{
  assert(false);
}

void FFMapper::map_copy(const MapperContext      ctx,
                        const Copy&              copy,
                        const MapCopyInput&      input,
                              MapCopyOutput&     output)
{
  assert(false);
}

void FFMapper::select_copy_sources(const MapperContext          ctx,
                                   const Copy&                  copy,
                                   const SelectCopySrcInput&    input,
                                         SelectCopySrcOutput&   output)
{
  assert(false);
}

void FFMapper::create_copy_temporary_instance(
                              const MapperContext              ctx,
                              const Copy&                      copy,
                              const CreateCopyTemporaryInput&  input,
                                    CreateCopyTemporaryOutput& output)
{
  assert(false);
}

void FFMapper::speculate(const MapperContext      ctx,
                         const Copy& copy,
                               SpeculativeOutput& output)
{
  assert(false);
}

void FFMapper::report_profiling(const MapperContext      ctx,
                                const Copy&              copy,
                                const CopyProfilingInfo& input)
{
  assert(false);
}

void FFMapper::select_sharding_functor(
                             const MapperContext                ctx,
                             const Copy&                        copy,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output)
{
  // Current all shardings uses data parallelism across machines
  output.chosen_functor = DataParallelShardingID;
}

void FFMapper::map_close(const MapperContext       ctx,
                         const Close&              close,
                         const MapCloseInput&      input,
                               MapCloseOutput&     output)
{
  assert(false);
}

void FFMapper::select_close_sources(const MapperContext        ctx,
                                    const Close&               close,
                                    const SelectCloseSrcInput&  input,
                                          SelectCloseSrcOutput& output)
{
  assert(false);
}

void FFMapper::create_close_temporary_instance(
                              const MapperContext               ctx,
                              const Close&                      close,
                              const CreateCloseTemporaryInput&  input,
                                    CreateCloseTemporaryOutput& output)
{
  assert(false);
}

void FFMapper::report_profiling(const MapperContext       ctx,
                                const Close&              close,
                                const CloseProfilingInfo& input)
{
  assert(false);
}

void FFMapper::select_sharding_functor(
                             const MapperContext                ctx,
                             const Close&                       close,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output)
{
  // Current all shardings uses data parallelism across machines
  output.chosen_functor = DataParallelShardingID;
}

void FFMapper::map_acquire(const MapperContext         ctx,
                           const Acquire&              acquire,
                           const MapAcquireInput&      input,
                                 MapAcquireOutput&     output)
{
  assert(false);
}

void FFMapper::speculate(const MapperContext         ctx,
                         const Acquire&              acquire,
                               SpeculativeOutput&    output)
{
  assert(false);
}

void FFMapper::report_profiling(const MapperContext         ctx,
                                const Acquire&              acquire,
                                const AcquireProfilingInfo& input)
{
  assert(false);
}

void FFMapper::select_sharding_functor(
                             const MapperContext                ctx,
                             const Acquire&                     acquire,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output)
{
  // Current all shardings uses data parallelism across machines
  output.chosen_functor = DataParallelShardingID;
}

void FFMapper::map_release(const MapperContext         ctx,
                           const Release&              release,
                           const MapReleaseInput&      input,
                                 MapReleaseOutput&     output)
{
  assert(false);
}

void FFMapper::select_release_sources(const MapperContext       ctx,
                                 const Release&                 release,
                                 const SelectReleaseSrcInput&   input,
                                       SelectReleaseSrcOutput&  output)
{
  assert(false);
}

void FFMapper::create_release_temporary_instance(
                               const MapperContext                 ctx,
                               const Release&                      release,
                               const CreateReleaseTemporaryInput&  input,
                                     CreateReleaseTemporaryOutput& output)
{
  assert(false);
}

void FFMapper::speculate(const MapperContext         ctx,
                         const Release&              release,
                               SpeculativeOutput&    output)
{
  assert(false);
}

void FFMapper::report_profiling(const MapperContext         ctx,
                                const Release&              release,
                                const ReleaseProfilingInfo& input)
{
  assert(false);
}

void FFMapper::select_sharding_functor(
                             const MapperContext                ctx,
                             const Release&                     release,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output)
{
  assert(false);
}

void FFMapper::select_partition_projection(const MapperContext  ctx,
                      const Partition&                          partition,
                      const SelectPartitionProjectionInput&     input,
                            SelectPartitionProjectionOutput&    output)
{
  assert(false);
}

void FFMapper::map_partition(const MapperContext        ctx,
                             const Partition&           partition,
                             const MapPartitionInput&   input,
                                   MapPartitionOutput&  output)
{
  assert(false);
}

void FFMapper::select_partition_sources(
                               const MapperContext             ctx,
                               const Partition&                partition,
                               const SelectPartitionSrcInput&  input,
                                     SelectPartitionSrcOutput& output)
{
  assert(false);
}

void FFMapper::create_partition_temporary_instance(
                          const MapperContext                   ctx,
                          const Partition&                      partition,
                          const CreatePartitionTemporaryInput&  input,
                                CreatePartitionTemporaryOutput& output)
{
  assert(false);
}

void FFMapper::report_profiling(const MapperContext              ctx,
                                const Partition&                 partition,
                                const PartitionProfilingInfo&    input)
{
  assert(false);
}

void FFMapper::select_sharding_functor(
                             const MapperContext                ctx,
                             const Partition&                   partition,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output)
{
  assert(false);
}

void FFMapper::select_sharding_functor(
                             const MapperContext                ctx,
                             const Fill&                        fill,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output)
{
  assert(false);
}

void FFMapper::configure_context(const MapperContext         ctx,
                                 const Task&                 task,
                                       ContextConfigOutput&  output)
{
  // Use the default values and do nothing
}

void FFMapper::select_tunable_value(const MapperContext         ctx,
                                    const Task&                 task,
                                    const SelectTunableInput&   input,
                                          SelectTunableOutput&  output)
{
  assert(false);
}

void FFMapper::select_sharding_functor(
                      const MapperContext                    ctx,
                      const MustEpoch&                       epoch,
                      const SelectShardingFunctorInput&      input,
                            MustEpochShardingFunctorOutput&  output)
{
  assert(false);
}

void FFMapper::map_must_epoch(const MapperContext           ctx,
                              const MapMustEpochInput&      input,
                                    MapMustEpochOutput&     output)
{
  // Directly assign each task to its target_proc
  for (unsigned i = 0; i < input.tasks.size(); i++) {
    output.task_processors[i] = input.tasks[i]->target_proc;
  }
  // Currently assume no constraints needed to be mapped
  assert(input.constraints.size() == 0);
}

void FFMapper::map_dataflow_graph(const MapperContext           ctx,
                                  const MapDataflowGraphInput&  input,
                                        MapDataflowGraphOutput& output)
{
  assert(false);
}

void FFMapper::memoize_operation(const MapperContext  ctx,
                                 const Mappable&      mappable,
                                 const MemoizeInput&  input,
                                       MemoizeOutput& output)
{
  // FIXME: Legion tracing currently does not support MUST_EPOCH
  if (mappable.as_must_epoch() != NULL) {
    output.memoize = false;
    return;
  }
  // Memoize all other mapping decisions
  output.memoize = true;
}

// Mapping control and stealing
void FFMapper::select_tasks_to_map(const MapperContext          ctx,
                                   const SelectMappingInput&    input,
                                         SelectMappingOutput&   output)
{
  // Just map all the ready tasks
  for (std::list<const Task*>::const_iterator it =
        input.ready_tasks.begin(); it != input.ready_tasks.end(); it++)
    output.map_tasks.insert(*it);
}

void FFMapper::select_steal_targets(const MapperContext         ctx,
                                    const SelectStealingInput&  input,
                                          SelectStealingOutput& output)
{
  // Nothing to do, no stealing in FFMapper
}

void FFMapper::permit_steal_request(const MapperContext         ctx,
                                    const StealRequestInput&    intput,
                                          StealRequestOutput&   output)
{
  // Nothing to do, no stealing in FFMapper
  assert(false);
}

//--------------------------------------------------------------------------
/*static*/ unsigned long long FFMapper::compute_task_hash(const Task &task)
//--------------------------------------------------------------------------
{
  // Use Sean's "cheesy" hash function
  const unsigned long long c1 = 0x5491C27F12DB3FA5; // big number, mix 1+0s
  const unsigned long long c2 = 353435097; // chosen by fair dice roll
  // We have to hash all region requirements including region names,
  // privileges, coherence modes, reduction operators, and fields
  unsigned long long result = c2 + task.task_id;
  for (unsigned idx = 0; idx < task.regions.size(); idx++)
  {
    const RegionRequirement &req = task.regions[idx];
    result = result * c1 + c2 + req.handle_type;
    if (req.handle_type != LEGION_PARTITION_PROJECTION) {
      result = result * c1 + c2 + req.region.get_tree_id();
      result = result * c1 + c2 + req.region.get_index_space().get_id();
      result = result * c1 + c2 + req.region.get_field_space().get_id();
    } else {
      result = result * c1 + c2 + req.partition.get_tree_id();
      result = result * c1 + c2 +
                              req.partition.get_index_partition().get_id();
      result = result * c1 + c2 + req.partition.get_field_space().get_id();
    }
    for (std::set<FieldID>::const_iterator it =
          req.privilege_fields.begin(); it !=
          req.privilege_fields.end(); it++)
      result = result * c1 + c2 + *it;
    result = result * c1 + c2 + req.privilege;
    result = result * c1 + c2 + req.prop;
    result = result * c1 + c2 + req.redop;
  }
  return result;
}


#ifdef DEADCODE
Memory FFMapper::default_policy_select_target_memory(MapperContext ctx,
                                                     Processor target_proc,
                                                     const RegionRequirement &req,
                                                     MemoryConstraint mc)
{
  if (target_proc.kind() == Processor::TOC_PROC) {
    if (req.tag == MAP_TO_ZC_MEMORY) {
      assert(proc_zcmems.find(target_proc) != proc_zcmems.end());
      return proc_zcmems[target_proc];
    } else {
      assert(req.tag == 0);
      //return DefaultMapper::default_policy_select_target_memory(
      //           ctx, target_proc, req, mc);
      assert(proc_fbmems.find(target_proc) != proc_fbmems.end());
      return proc_fbmems[target_proc];
    }
  } else if (target_proc.kind() == Processor::LOC_PROC) {
    assert(proc_zcmems.find(target_proc) != proc_zcmems.end());
    return proc_zcmems[target_proc];
  } else {
    return DefaultMapper::default_policy_select_target_memory(
               ctx, target_proc, req, mc);
  }
}
#endif

const std::vector<Processor>& FFMapper::all_procs_by_kind(Processor::Kind kind)
{
  switch (kind)
  {
    case Processor::LOC_PROC:
      return all_cpus;
    case Processor::TOC_PROC:
      return all_gpus;
    case Processor::PY_PROC:
      return all_pys;
    default:
      assert(0);
  }
  return all_cpus;
}

Memory FFMapper::default_select_target_memory(
    MapperContext ctx,
    Processor target_proc,
    const RegionRequirement &req)
{
  if (target_proc.kind() == Processor::TOC_PROC) {
    if (req.tag == MAP_TO_ZC_MEMORY) {
      assert(proc_zcmems.find(target_proc) != proc_zcmems.end());
      return proc_zcmems[target_proc];
    } else {
      assert(req.tag == 0);
      assert(proc_fbmems.find(target_proc) != proc_fbmems.end());
      return proc_fbmems[target_proc];
    }
  } else if (target_proc.kind() == Processor::LOC_PROC) {
    assert(proc_zcmems.find(target_proc) != proc_zcmems.end());
    return proc_zcmems[target_proc];
  } else if (target_proc.kind() == Processor::PY_PROC) {
    assert(proc_zcmems.find(target_proc) != proc_zcmems.end());
    return proc_zcmems[target_proc];
  } else {
    // Unknown processor kind
    assert(false);
    return Memory::NO_MEMORY;
  }
}

bool FFMapper::default_make_instance(
    MapperContext ctx, Memory target_mem,
         const LayoutConstraintSet &constraints,
         PhysicalInstance &result,
         bool meets_constraints,
         const RegionRequirement &req,
         size_t *footprint)
{
  LogicalRegion target_region = req.region;
  bool tight_region_bounds = false;
  bool created = true;
  std::vector<LogicalRegion> target_regions(1, target_region);
  if (!runtime->find_or_create_physical_instance(ctx, target_mem, constraints,
      target_regions, result, created, true/*acquire*/, 0/*priority*/,
      tight_region_bounds, footprint))
    return false;
  if (created) {
    int priority = LEGION_GC_NEVER_PRIORITY;
    if (priority != 0)
      runtime->set_garbage_collection_priority(ctx, result, priority);
  }
  return true;
}

LayoutConstraintID FFMapper::default_select_layout_constraints(
         MapperContext ctx, Memory target_memory,
         const RegionRequirement &req,
         bool needs_field_constraint_check)
{
  assert(req.privilege != LEGION_REDUCE);
  std::pair<Memory::Kind,FieldSpace> constraint_key(target_memory.kind(),
      req.region.get_field_space());
  std::map<std::pair<Memory::Kind,FieldSpace>,LayoutConstraintID>::
      const_iterator finder = layout_constraint_cache.find(constraint_key);
  if (finder != layout_constraint_cache.end())
  {
    // If we don't need a constraint check we are already good
    if (!needs_field_constraint_check)
      return finder->second;
    // Check that the fields still are the same, if not, fall through
    // so that we make a new set of constraints
    const LayoutConstraintSet &old_constraints =
            runtime->find_layout_constraints(ctx, finder->second);
    // Should be only one unless things have changed
    const std::vector<FieldID> &old_set =
                      old_constraints.field_constraint.get_field_set();
    // Check to make sure the field sets are still the same
    std::vector<FieldID> new_fields;
    runtime->get_field_space_fields(ctx,
                                    constraint_key.second,new_fields);
    if (new_fields.size() == old_set.size())
    {
      std::set<FieldID> old_fields(old_set.begin(), old_set.end());
      bool still_equal = true;
      for (unsigned idx = 0; idx < new_fields.size(); idx++)
      {
        if (old_fields.find(new_fields[idx]) == old_fields.end())
        {
          still_equal = false;
          break;
        }
      }
      if (still_equal)
        return finder->second;
    }
    // Otherwise we fall through and make a new constraint which
    // will also update the cache
  }
  // Fill in the constraints
  LayoutConstraintSet constraints;
  default_select_constraints(ctx, constraints, target_memory, req);
  // Do the registration
  LayoutConstraintID result =
    runtime->register_layout(ctx, constraints);
  // Record our results, there is a benign race here as another mapper
  // call could have registered the exact same registration constraints
  // here if we were preempted during the registration call. The
  // constraint sets are identical though so it's all good.
  layout_constraint_cache[constraint_key] = result;
  return result;
}

void FFMapper::default_select_constraints(
         MapperContext ctx,
         LayoutConstraintSet &constraints,
         Memory target_memory,
         const RegionRequirement &req)
{
  // Currently don't support reduction instance
  assert(req.privilege != LEGION_REDUCE);
  // Our base default mapper will try to make instances of containing
  // all fields (in any order) laid out in SOA format to encourage
  // maximum re-use by any tasks which use subsets of the fields
  constraints.add_constraint(SpecializedConstraint())
    .add_constraint(MemoryConstraint(target_memory.kind()));

  if (constraints.field_constraint.field_set.size() == 0)
  {
    // Normal instance creation
    std::vector<FieldID> fields;
    FieldSpace handle = req.region.get_field_space();
    runtime->get_field_space_fields(ctx, handle, fields);
    // Currently assume each tensor has exactly one field
    assert(fields.size() == 1);
    constraints.add_constraint(FieldConstraint(fields,false/*contiguous*/,
                                               false/*inorder*/));
  }
  if (constraints.ordering_constraint.ordering.size() == 0)
  {
    IndexSpace is = req.region.get_index_space();
    Domain domain = runtime->get_index_space_domain(ctx, is);
    int dim = domain.get_dim();
    std::vector<DimensionKind> dimension_ordering(dim + 1);
    for (int i = 0; i < dim; ++i)
      dimension_ordering[i] =
        static_cast<DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
    dimension_ordering[dim] = LEGION_DIM_F;
    constraints.add_constraint(OrderingConstraint(dimension_ordering,
                                                  false/*contigous*/));
  }
}


void update_mappers(Machine machine, Runtime *runtime,
                    const std::set<Processor> &local_procs)
{

/*
  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];
    if (affinity.p.kind() == Processor::TOC_PROC) {
      if (affinity.m.kind() == Memory::GPU_FB_MEM) {
        (*proc_fbmems)[affinity.p] = affinity.m;
      }
      else if (affinity.m.kind() == Memory::Z_COPY_MEM) {
        (*proc_zcmems)[affinity.p] = affinity.m;
      }
    }
  }
  for (std::map<Processor, Memory>::iterator it = proc_fbmems->begin();
       it != proc_fbmems->end(); it++) {
    gpus->push_back(it->first);
  }
*/
  // Find strategy file path
  std::string strategyFile = "";
  const InputArgs &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  bool enable_control_replication = false;
  for (int i = 1; i < argc; i++) {
    if ((!strcmp(argv[i], "--import")) || (!strcmp(argv[i], "--import-strategy"))) {
      strategyFile = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--control-replication")) {
      enable_control_replication = true;
      continue;
    }
  }

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    FFMapper* mapper = new FFMapper(runtime->get_mapper_runtime(),
                                    machine, *it, "FlexFlow Mapper",
                                    strategyFile, enable_control_replication);
    runtime->replace_default_mapper(mapper, *it);
  }
}
