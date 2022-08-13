/**
 * @file memory_optimization.h
 * @brief Memory optimization related stuff
 *
 * @copyright Copyright 2022 CMU
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

#pragma once

#include <string>
#include <cassert>

namespace FlexFlow {

enum class MemoryUsageType {
  // Use global memory of a PCG as the measure of memory usage. No device
  // mapping consideration.
  GLOBAL,

  // Use the max of peak per-device memory usage among devices as the measure.
  // Need associated device mapping views.
  PER_DEVICE_MAX,

  // Use detailed per-device memory usage as the measure. Need associated device
  // mapping views.
  PER_DEVICE_ALL,
};

enum class MemorySearchAlgo {
  // Multiple objective DP search. Combine memory cost and run time cost into
  // one single cost function and add a factor to balance them.
  MULTI_OBJECTIVE,
};

/**
 * @brief Config class to control memory optimizations. This should be put into
 * config.h and be stored in FFConfig. But for easy turnaround, put this here
 * for now.
 */
class MemoryOptimConfig {
public:
  MemoryUsageType mem_usage_type;   ///< How to represent memory cost
  MemorySearchAlgo mem_search_algo; ///< How to search for the optimal schedule
  float run_time_cost_factor; ///< The weight factor of run time cost in the
                              ///< overall cost function; used in
                              ///< MULTI_OBJECTIVE algorithm

  MemoryOptimConfig()
      : mem_usage_type{MemoryUsageType::GLOBAL},
        mem_search_algo{MemorySearchAlgo::MULTI_OBJECTIVE},
        run_time_cost_factor{0.5} {}
  MemoryOptimConfig(float factor)
      : mem_usage_type{MemoryUsageType::GLOBAL},
        mem_search_algo{MemorySearchAlgo::MULTI_OBJECTIVE},
        run_time_cost_factor{factor} {}
};

namespace PCG {

/**
 * @brief Class to hold memory usage information of a (sub-)PCG.
 */
class MemoryUsage {
public:
  MemoryUsageType usage_type; ///< What "num" means
  float num;                  ///< The numerical number of memory usage

  // May need this in the future, but not for now.
  // std::vector<float> nums;     ///< Detailed number of usage for all devices

  ///
  /// Public APIs
  ///
  MemoryUsage() : usage_type{MemoryUsageType::GLOBAL}, num{0.0} {}
  MemoryUsage(MemoryUsageType _usage_type, float _num)
      : usage_type{_usage_type}, num{_num} {}

  std::string to_string() const {
    std::string type_name;
    switch (usage_type) {
      case MemoryUsageType::GLOBAL:
        type_name = "GLOBAL";
        break;
      case MemoryUsageType::PER_DEVICE_MAX:
        type_name = "PER_DEVICE_MAX";
        break;
      case MemoryUsageType::PER_DEVICE_ALL:
        // Not supporting detailed per-device memory usage now.
        assert(false);
        break;
    }
    return "(MemoryUsageType:" + type_name + ", Usage:" + std::to_string(num) +
           ")";
  }

  MemoryUsage &operator+=(MemoryUsage const &rhs) {
    assert(usage_type == rhs.usage_type);

    // Handle the merge of memory usage differently here.
    switch (usage_type) {
      case MemoryUsageType::GLOBAL:
        num += rhs.num;
        break;
      case MemoryUsageType::PER_DEVICE_MAX:
        num = std::max(num, rhs.num);
        break;
      case MemoryUsageType::PER_DEVICE_ALL:
        // Not supporting detailed per-device memory usage now.
        assert(false);
        break;
    }

    return *this;
  }

  /**
   * @brief Combine the memory usage of two PCGs flexibly based on
   * MemoryUsageType.
   */
  friend MemoryUsage operator+(MemoryUsage lhs, MemoryUsage const &rhs) {
    lhs += rhs;
    return lhs;
  }

  friend std::ostream &operator<<(std::ostream &s, MemoryUsage const &usage) {
    s << usage.to_string();
    return s;
  }
};

/**
 * @brief The choice of memory optimizations applied to a Graph.
 */
class MemOptDecision {
public:
private:
};

} // namespace PCG
} // namespace FlexFlow
