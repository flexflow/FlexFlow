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

#ifndef __RNN_MAPPER_H__
#define __RNN_MAPPER_H__

#include "default_mapper.h"
#include "legion.h"
#include "ops.h"

using namespace Legion;
using namespace Legion::Mapping;

class RnnMapper : public DefaultMapper {
public:
  RnnMapper(MapperRuntime *rt,
            Machine machine,
            Processor local,
            char const *mapper_name,
            std::vector<Processor> *gpus,
            std::map<Processor, Memory> *proc_fbmems,
            std::vector<Processor> *cpus);

public:
  virtual void select_task_options(const MapperContext ctx,
                                   Task const &task,
                                   TaskOptions &output);
  // virtual void slice_task(const MapperContext ctx,
  //                       const Task& task,
  //                     const SliceTaskInput& input,
  //                   SliceTaskOutput& output);
  // virtual void map_task(const MapperContext ctx,
  //                     const Task& task,
  //                   const MapTaskInput& input,
  //                 MapTaskOutput& output);
  // virtual void select_task_sources(const MapperContext ctx,
  //                                const Task& task,
  //                              const SelectTaskSrcInput& input,
  //                            SelectTaskSrcOutput& output);
  static MappingTagID assign_to_gpu(int gpuIdx);

protected:
  std::vector<Processor> &gpus;
  std::map<Processor, Memory> &proc_fbmems;
  std::vector<Processor> &cpus;
};

void update_mappers(Machine machine,
                    Runtime *rt,
                    std::set<Processor> const &local_procs);
#endif
