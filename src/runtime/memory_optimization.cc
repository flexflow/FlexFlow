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

#include "flexflow/memory_optimization.h"

namespace FlexFlow {

namespace PCG {

std::string MemoryUsage::to_string() const {
  std::string type_name;
  switch (usage_type) {
    case MemoryUsageType::GLOBAL:
      type_name = "GLOBAL";
      break;
    case MemoryUsageType::PER_DEVICE_MAX:
      type_name = "PER_DEVICE_MAX";
      break;
  }
  return "(MemoryUsageType:" + type_name + ", Usage:" + std::to_string(num) +
         ")";
}

MemoryUsage &MemoryUsage::operator+=(MemoryUsage const &rhs) {
  assert(usage_type == rhs.usage_type);

  // Handle the merge of memory usage differently here.
  switch (usage_type) {
    case MemoryUsageType::GLOBAL:
      num += rhs.num;
      break;
    case MemoryUsageType::PER_DEVICE_MAX:
      num = std::max(num, rhs.num);
      break;
  }

  return *this;
}

MemoryUsage operator+(MemoryUsage lhs, MemoryUsage const &rhs) {
  lhs += rhs;
  return lhs;
}

std::ostream &operator<<(std::ostream &s, MemoryUsage const &usage) {
  s << usage.to_string();
  return s;
}

} // namespace PCG

} // namespace FlexFlow
