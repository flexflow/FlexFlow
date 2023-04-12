/* Copyright 2022 NVIDIA CORPORATION
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

#include "strategy.h"
#include <memory>

namespace triton { namespace backend { namespace legion {

//
// Mock implementation of the class with synthetic state,
// will raise error if trying to invoke unintended methods
//
LayerStrategy::LayerStrategy(
    Legion::ShardingID sid, Legion::MappingTagID tag, Legion::Runtime* runtime)
    : kind(Realm::Processor::Kind::LOC_PROC), sharding_function(nullptr),
      tag(tag)
{
}

LayerStrategy::~LayerStrategy() {}

bool
LayerStrategy::is_local_processor(Realm::Processor proc) const
{
  for (size_t i = 0; i < nProcs; ++i) {
    if (local_processors[i] == proc) {
      return true;
    }
  }
  return false;
}

unsigned
LayerStrategy::find_local_offset(Realm::Processor proc) const
{
  for (unsigned idx = 0; idx < nProcs; idx++)
    if (local_processors[idx] == proc)
      return idx;
  throw std::invalid_argument("Getting offset for a non-local processor");
}

std::unique_ptr<LayerStrategy>
CreateMockLayerStrategy(
    const std::vector<Realm::Processor>& local_processors,
    const std::vector<Realm::Processor>& global_processors)
{
  std::unique_ptr<LayerStrategy> ls(new LayerStrategy(0, 0, nullptr));
  ls->nProcs = local_processors.size();
  for (size_t i = 0; i < local_processors.size(); ++i) {
    ls->local_processors[i] = local_processors[i];
  }
  ls->global_processors = global_processors;
  return ls;
}

PartitionStrategy::~PartitionStrategy() {}

}}}  // namespace triton::backend::legion
