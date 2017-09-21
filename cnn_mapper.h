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
#include "ops.h"

using namespace Legion;
using namespace Legion::Mapping;

class CnnMapper : public DefaultMapper {
public:
  CnnMapper(MapperRuntime *rt, Machine machine, Processor local,
            const char *mapper_name, std::vector<Processor>* gpus,
            std::map<Processor, Memory>* proc_fbmems);
public:
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output);
  virtual void map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                        MapTaskOutput& output);
protected:
  std::vector<Processor>& gpus;
  std::map<Processor, Memory>& proc_fbmems;
};

void update_mappers(Machine machine, Runtime *rt, const std::set<Processor> &local_procs);
#endif
