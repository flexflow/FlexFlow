/* Copyright 2019 Stanford University
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
                   const char *mapper_name,
                   std::vector<Processor>* _gpus,
                   std::map<Processor, Memory>* _proc_fbmems,
                   std::map<Processor, Memory>* _proc_zcmems,
                   std::vector<Processor>* _cpus,
                   std::map<size_t, ParallelConfig>* _strategies)
  : DefaultMapper(rt, machine, local, mapper_name),
    gpus(*_gpus), proc_fbmems(*_proc_fbmems),
    proc_zcmems(*_proc_zcmems), cpus(*_cpus),
    strategies(*_strategies)
{}

void FFMapper::slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output)
{
  //printf("task.task_id = %d task.target_proc = %x num_slices = %zu gpus.size = %zu\n",
  //    task.task_id, task.target_proc.id, input.domain.get_volume(), gpus.size());
  if ((task.task_id == TOP_LEVEL_TASK_ID)
  || ((task.task_id >= CUSTOM_CPU_TASK_ID_FIRST)
     && (task.task_id <= CUSTOM_CPU_TASK_ID_LAST))) {
    DefaultMapper::slice_task(ctx, task, input, output);
  } else {
    output.slices.resize(input.domain.get_volume());
    MappingTagID hash = task.tag;
    // Make sure the task has a non-zero tag
    assert(hash != 0);
    ParallelConfig config;
    unsigned int config_num_parts = 1;
    if (strategies.find(hash) == strategies.end()) {
      // No strategy found, use default data parallelism
      int ndim = input.domain.get_dim();
      assert(strategies.find(FFConfig::DataParallelism_1D-1+ndim) != strategies.end());
      config = strategies[FFConfig::DataParallelism_1D-1+ndim];
    } else {
      // Found a strategy
      config = strategies[hash];
      // Check that the dimensions match
      assert(config.nDims == input.domain.get_dim());
    }
    for (int i = 0; i < config.nDims; i++) {
      //assert(config.dim[i] == input.domain.hi()[i] - input.domain.lo()[i] + 1);
      config_num_parts *= config.dim[i];
    }
    const std::vector<Processor>* devices;
    if (config.device_type == ParallelConfig::GPU) {
      devices = &gpus;
    } else {
      devices = &cpus;
    }
    switch (input.domain.get_dim())
    {
      case 1:
      {
        Rect<1> rect = input.domain;
        int cnt = 0;
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
          unsigned int idx = 0;
          for (int i = input.domain.get_dim()-1; i >= 0; i--)
            idx = idx*(input.domain.hi()[i]-input.domain.lo()[i]+1)+pir[i];
          assert(config_num_parts > idx);
          //assert((int)gpus.size() > config.gpu[idx]);
          Rect<1> slice(*pir, *pir);
          output.slices[cnt++] = TaskSlice(slice,
              (*devices)[config.device_ids[idx] % devices->size()],
              false/*recurse*/, false/*stealable*/);
        }
        break;
      }
      case 2:
      {
        Rect<2> rect = input.domain;
        int cnt = 0;
        for (PointInRectIterator<2> pir(rect); pir(); pir++) {
          unsigned int idx = 0;
          for (int i = input.domain.get_dim()-1; i >= 0; i--)
            idx = idx*(input.domain.hi()[i]-input.domain.lo()[i]+1)+pir[i];
          assert(config_num_parts > idx);
          //assert((int)gpus.size() > config.gpu[idx]);
          Rect<2> slice(*pir, *pir);
          output.slices[cnt++] = TaskSlice(slice,
              (*devices)[config.device_ids[idx] % devices->size()],
              false/*recurse*/, false/*stealable*/);
        }
        break;
      }
      case 3:
      {
        Rect<3> rect = input.domain;
        int cnt = 0;
        for (PointInRectIterator<3> pir(rect); pir(); pir++) {
          unsigned int idx = 0;
          for (int i = input.domain.get_dim()-1; i >= 0; i--)
            idx = idx*(input.domain.hi()[i]-input.domain.lo()[i]+1)+pir[i];
          assert(config_num_parts > idx);
          //assert((int)gpus.size() > config.gpu[idx]);
          Rect<3> slice(*pir, *pir);
          output.slices[cnt++] = TaskSlice(slice,
              (*devices)[config.device_ids[idx] % devices->size()],
              false/*recurse*/, false/*stealable*/);
        }
        break;
      }
      case 4:
      {
        Rect<4> rect = input.domain;
        int cnt = 0;
        for (PointInRectIterator<4> pir(rect); pir(); pir++) {
          unsigned int idx = 0;
          for (int i = input.domain.get_dim()-1; i >= 0; i--)
            idx = idx*(input.domain.hi()[i]-input.domain.lo()[i]+1)+pir[i];
          assert(config_num_parts > idx);
          //assert((int)gpus.size() > config.gpu[idx]);
          Rect<4> slice(*pir, *pir);
          output.slices[cnt++] = TaskSlice(slice,
              (*devices)[config.device_ids[idx] % devices->size()],
              false/*recurse*/, false/*stealable*/);
        }
        break;
      }
      default:
        assert(false);
    }
  }
}

void FFMapper::select_task_options(const MapperContext ctx,
                                   const Task& task,
                                   TaskOptions& output)
{
  unsigned long long task_hash = compute_task_hash(task);
  if (task.task_id == SGD_UPD_TASK_ID) {
    // For SGD Update, pick a processor from config
    // TODO: perform similar optimizations for other Optimizer
    MappingTagID hash = task.tag;
    ParallelConfig config;
    if (strategies.find(hash) != strategies.end()) {
      config = strategies[hash];
      int num_parts = 1;
      for (int i = 0; i < config.nDims; i++)
        num_parts *= config.dim[i];
      if (num_parts == 1) {
        output.initial_proc = gpus[config.device_ids[0]];
        output.inline_task = false;
        output.stealable = stealing_enabled;
        output.map_locally = map_locally;
        return;
      }
    }
    if (cache_update_tasks.find(task_hash) != cache_update_tasks.end()) {
      output.initial_proc = cache_update_tasks[task_hash];
      output.inline_task = false;
      output.stealable = stealing_enabled;
      output.map_locally = map_locally;
      return;
    }
  }

  if (task.task_id == STRATEGY_SEARCH_TASK_ID) {
    output.initial_proc = gpus[0];
    output.inline_task = false;
    output.stealable = stealing_enabled;
    output.map_locally = map_locally;
    return;
  }
  
  DefaultMapper::select_task_options(ctx, task, output);
  if ((task.task_id == SGD_UPD_TASK_ID)
  && (cache_update_tasks.find(task_hash) == cache_update_tasks.end())) {
    cache_update_tasks[task_hash] = output.initial_proc;
    //printf("hash = %llu proc = %llu\n", task_hash, output.initial_proc.id);
  }
}

void FFMapper::select_sharding_functor(const MapperContext ctx,
                                       const Task& task,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output)
{
  // Current all shardings uses data parallelism across machines
  output.chosen_functor = DataParallelShardingID;
}

Memory FFMapper::default_policy_select_target_memory(MapperContext ctx,
                                                     Processor target_proc,
                                                     const RegionRequirement &req)
{
  if (target_proc.kind() == Processor::TOC_PROC) {
    if (req.tag == MAP_TO_ZC_MEMORY) {
      assert(proc_zcmems.find(target_proc) != proc_zcmems.end());
      return proc_zcmems[target_proc];
    } else {
      assert(req.tag == 0);
      //return DefaultMapper::default_policy_select_target_memory(
      //           ctx, target_proc, req);
      assert(proc_fbmems.find(target_proc) != proc_fbmems.end());
      return proc_fbmems[target_proc];
    }
  } else if (target_proc.kind() == Processor::LOC_PROC) {
    assert(proc_zcmems.find(target_proc) != proc_zcmems.end());
    return proc_zcmems[target_proc];
  } else {
    return DefaultMapper::default_policy_select_target_memory(
               ctx, target_proc, req);
  }
}

void FFMapper::map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                        MapTaskOutput& output)
{
  // Convolve forward
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
    DefaultMapper::map_task(ctx, task, input, output);
}

void update_mappers(Machine machine, Runtime *runtime,
                    const std::set<Processor> &local_procs)
{
  std::vector<Processor>* gpus = new std::vector<Processor>();
  std::map<Processor, Memory>* proc_fbmems = new std::map<Processor, Memory>();
  std::map<Processor, Memory>* proc_zcmems = new std::map<Processor, Memory>();
  std::vector<Processor>* cpus = new std::vector<Processor>();
  //std::map<Processor, Memory>* proc_zcmems = new std::map<Processor, Memory>();
  std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);
  Machine::ProcessorQuery proc_query(machine);
  for (Machine::ProcessorQuery::iterator it = proc_query.begin();
      it != proc_query.end(); it++)
  {
    if (it->kind() == Processor::TOC_PROC) {
      gpus->push_back(*it);
      Machine::MemoryQuery fb_query(machine);
      fb_query.only_kind(Memory::GPU_FB_MEM);
      fb_query.best_affinity_to(*it);
      assert(fb_query.count() == 1);
      (*proc_fbmems)[*it] = *(fb_query.begin());
      Machine::MemoryQuery zc_query(machine);
      zc_query.only_kind(Memory::Z_COPY_MEM);
      zc_query.has_affinity_to(*it);
      assert(zc_query.count() == 1);
      (*proc_zcmems)[*it] = *(zc_query.begin());
    }
    else if (it->kind() == Processor::LOC_PROC) {
      cpus->push_back(*it);
      Machine::MemoryQuery zc_query(machine);
      zc_query.only_kind(Memory::Z_COPY_MEM);
      zc_query.has_affinity_to(*it);
      assert(zc_query.count() == 1);
      (*proc_zcmems)[*it] = *(zc_query.begin());
    }
  }

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
  for (int i = 1; i < argc; i++) {
    if ((!strcmp(argv[i], "-s")) || (!strcmp(argv[i], "--strategy"))) {
      strategyFile = std::string(argv[++i]);
      continue;
    }
  }
  std::map<MappingTagID, ParallelConfig>* strategies = new std::map<MappingTagID, ParallelConfig>();

  if (strategyFile == "") {
    // No strategy file provided, use data parallelism
    log_ff_mapper.print("No strategy file provided. Use default data parallelism.");
  } else {
    log_ff_mapper.print("Load parallelization strategy from file %s",
                     strategyFile.c_str());
    load_strategies_from_file(strategyFile, *strategies);
  }
  for (int i = FFConfig::DataParallelism_1D; i <= FFConfig::DataParallelism_4D; i++) {
    ParallelConfig pc;
    pc.device_type = ParallelConfig::GPU;
    pc.nDims = i - FFConfig::DataParallelism_1D + 1;
    for (int j = 0; j < pc.nDims; j++)
      pc.dim[j] = 1;
    pc.dim[pc.nDims-1] = gpus->size();
    for (size_t j = 0; j < gpus->size(); j++)
      pc.device_ids[j] = j;
    (*strategies)[i] = pc;
  }

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    FFMapper* mapper = new FFMapper(runtime->get_mapper_runtime(),
                                    machine, *it, "FlexFlow Mapper",
                                    gpus, proc_fbmems, proc_zcmems,
                                    cpus, strategies);
    runtime->replace_default_mapper(mapper, *it);
  }
}

