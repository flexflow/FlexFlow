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

#ifndef _FLEXFLOW_MEMORY_OPTIMIZATION_H_
#define _FLEXFLOW_MEMORY_OPTIMIZATION_H_

#include <cassert>
#include <string>

namespace FlexFlow {

enum class MemoryUsageType {
  // Use global memory of a PCG as the measure of memory usage. No device
  // mapping consideration.
  GLOBAL,

  // Use the max of peak per-device memory usage among devices as the measure.
  // Need associated device mapping views.
  PER_DEVICE_MAX,
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
                              ///< Valid between and including 0 and 1

  MemoryOptimConfig()
      : mem_usage_type{MemoryUsageType::GLOBAL},
        mem_search_algo{MemorySearchAlgo::MULTI_OBJECTIVE},
        run_time_cost_factor{0.5} {}
  MemoryOptimConfig(float factor)
      : mem_usage_type{MemoryUsageType::GLOBAL},
        mem_search_algo{MemorySearchAlgo::MULTI_OBJECTIVE},
        run_time_cost_factor{factor} {}
};

/**
 * @brief Hold the result (including memory information) of a graph_optimize on
 * a PCG.
 */
class MemorySearchResult {
public:
  float run_time_cost{};
  float memory_cost{};
  float search_time{};
  ///< The max of per-device memory usage among all devices
  float max_per_device_mem_all_deivces = 0.0;
};

namespace PCG {

/**
 * @brief Class to hold memory usage information of a (sub-)PCG.
 */
class MemoryUsage {
public:
  MemoryUsageType usage_type; ///< What "num" means
  float num;                  ///< The numerical number of memory usage

  MemoryUsage() : usage_type{MemoryUsageType::GLOBAL}, num{0.0} {}
  MemoryUsage(MemoryUsageType _usage_type, float _num)
      : usage_type{_usage_type}, num{_num} {}

  std::string to_string() const;

  MemoryUsage &operator+=(MemoryUsage const &rhs);

  /**
   * @brief Combine the memory usage of two PCGs flexibly based on
   * MemoryUsageType.
   */
  friend MemoryUsage operator+(MemoryUsage lhs, MemoryUsage const &rhs);

  friend std::ostream &operator<<(std::ostream &s, MemoryUsage const &usage);
};

} // namespace PCG
} // namespace FlexFlow

#endif // _FLEXFLOW_MEMORY_OPTIMIZATION_H_
