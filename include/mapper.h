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
#include "model.h"

using namespace Legion;
using namespace Legion::Mapping;

class FFMapper : public DefaultMapper {
public:
  FFMapper(MapperRuntime *rt, Machine machine, Processor local,
            const char *mapper_name, std::vector<Processor>* gpus,
            std::map<Processor, Memory>* proc_fbmems,
            std::map<Processor, Memory>* proc_zcmems,
            std::vector<Processor>* cpus,
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

void update_mappers(Machine machine, Runtime *rt, const std::set<Processor> &local_procs);
#endif
