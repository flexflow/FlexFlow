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

#include "mapper.h"

LegionRuntime::Logger::Category log_mapper("Mapper");

FFMapper::FFMapper(MapperRuntime *rt, Machine machine, Processor local,
                     const char *mapper_name,
                     std::vector<Processor>* _gpus,
                     std::map<Processor, Memory>* _proc_fbmems,
                     std::vector<Processor>* _cpus,
                     std::map<size_t, ParallelConfig>* _strategies)
  : DefaultMapper(rt, machine, local, mapper_name),
    gpus(*_gpus), proc_fbmems(*_proc_fbmems), cpus(*_cpus),
    strategies(*_strategies)
{}

void FFMapper::slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output)
{
#ifdef DEADCODE
  if (task.task_id == LOAD_IMAGES_TASK_ID) {
    output.slices.resize(input.domain.get_volume());
    unsigned idx = 0;
    assert(input.domain.get_dim() == 1);
    Rect<1> rect = input.domain;
    for (PointInRectIterator<1> pir(rect); pir(); pir++, idx++) {
      Rect<1> slice(*pir, *pir);
      output.slices[idx] = TaskSlice(slice, cpus[idx % cpus.size()],
                                     false/*recurse*/, false/*stealable*/);
    }
  }
  else 
#endif
  if (task.task_id != TOP_LEVEL_TASK_ID)
  {
    output.slices.resize(input.domain.get_volume());
    unsigned idx = 0;
    MappingTagID hash = task.tag;
    // Make sure the task has a non-zero tag
    assert(hash != 0);
    ParallelConfig config;
    if (strategies.find(hash) == strategies.end()) {
      // No strategy found, use default data parallelism
      assert(strategies.find(FFConfig::DataParallelismID) != strategies.end());
      config = strategies[FFConfig::DataParallelismID];
    } else {
      // Found a strategy
      config = strategies[hash];
      // Check that the dimensions match
      assert(config.nDims == input.domain.get_dim());
      for (int i = 0; i < config.nDims; i++)
        assert(config.dim[i] == input.domain.hi()[i] - input.domain.lo()[i] + 1);
    }
    switch (input.domain.get_dim())
    {
      case 1:
      {
        Rect<1> rect = input.domain;
        for (PointInRectIterator<1> pir(rect); pir(); pir++, idx++) {
          Rect<1> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice, gpus[config.gpu[idx]],
                                         false/*recurse*/, false/*stealable*/);
        }
        break;
      }
      case 2:
      {
        Rect<2> rect = input.domain;
        for (PointInRectIterator<2> pir(rect); pir(); pir++, idx++) {
          Rect<2> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice, gpus[config.gpu[idx]],
                                         false/*recurse*/, false/*stealable*/);
        }
        break;
      }
      case 3:
      {
        Rect<3> rect = input.domain;
        for (PointInRectIterator<3> pir(rect); pir(); pir++, idx++) {
          Rect<3> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice, gpus[config.gpu[idx]],
                                         false/*recurse*/, false/*stealable*/);
        }
        break;
      }
      default:
        assert(false);
    }
  }
  else
    DefaultMapper::slice_task(ctx, task, input, output);
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
  }
  else
    DefaultMapper::map_task(ctx, task, input, output);
}

void update_mappers(Machine machine, Runtime *runtime,
                    const std::set<Processor> &local_procs)
{
  std::vector<Processor>* gpus = new std::vector<Processor>();
  std::map<Processor, Memory>* proc_fbmems = new std::map<Processor, Memory>();
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
    }
    else if (it->kind() == Processor::LOC_PROC) {
      cpus->push_back(*it);
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
    log_mapper.print("No strategy file provided. Use default data parallelism.");
    // No strategy file provided, use data parallelism
    // TODO: the decault data parallelsim only apply to 2D operators
    ParallelConfig pc;
    pc.nDims = 2;
    pc.dim[0] = 1;
    pc.dim[1] = gpus->size();
    for (size_t i = 0; i < gpus->size(); i++)
      pc.gpu[i] = i;
    (*strategies)[FFConfig::DataParallelismID] = pc;
  } else {
    load_strategies_from_file(strategyFile, *strategies);
  }

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    FFMapper* mapper = new FFMapper(runtime->get_mapper_runtime(),
                                    machine, *it, "FlexFlow Mapper",
                                    gpus, proc_fbmems, cpus, strategies);
    runtime->replace_default_mapper(mapper, *it);
  }
}

