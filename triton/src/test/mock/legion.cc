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

#include "legion.h"
#include <string>

namespace Legion {

//
// Dummy implementation that will raise error if trying to invoke methods
//
const IndexSpace IndexSpace::NO_SPACE = IndexSpace();
const FieldSpace FieldSpace::NO_SPACE = FieldSpace();
const IndexPartition IndexPartition::NO_PART = IndexPartition();

FieldSpace::FieldSpace() : id(0) {}
IndexSpace::IndexSpace() : id(0), tid(0), type_tag(0) {}
IndexPartition::IndexPartition() : id(0), tid(0), type_tag(0) {}
LogicalPartition::LogicalPartition()
    : tree_id(0), index_partition(IndexPartition::NO_PART),
      field_space(FieldSpace::NO_SPACE)
{
}
LogicalRegion::LogicalRegion()
    : tree_id(0), index_space(IndexSpace::NO_SPACE),
      field_space(FieldSpace::NO_SPACE)
{
}
const LogicalRegion LogicalRegion::NO_REGION = LogicalRegion();
const LogicalPartition LogicalPartition::NO_PART = LogicalPartition();

Predicate::Predicate() {}
Predicate::~Predicate() {}
Grant::Grant() {}
Grant::~Grant() {}
Future::Future() {}
Future::~Future() {}
FutureMap::FutureMap() {}
FutureMap::~FutureMap() {}
ArgumentMap::ArgumentMap() {}
ArgumentMap::~ArgumentMap() {}
RegionRequirement::RegionRequirement() {}
RegionRequirement::~RegionRequirement() {}
IndexTaskLauncher::IndexTaskLauncher() {}
ExternalResources::ExternalResources() {}
ExternalResources::~ExternalResources() {}
PhysicalRegion::PhysicalRegion() {}
PhysicalRegion::~PhysicalRegion() {}

}  // namespace Legion
