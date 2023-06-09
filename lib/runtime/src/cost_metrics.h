#ifndef _FLEXFLOW_RUNTIME_SRC_COST_METRICS_H
#define _FLEXFLOW_RUNTIME_SRC_COST_METRICS_H

#include "utils/visitable.h"

namespace FlexFlow {

/**
 * @brief Costs of an operator.
 */
struct CostMetrics : public use_visitable_cmp<CostMetrics> {
  CostMetrics() = delete;
  CostMetrics(float forward_time,
              float backward_time,
              float sync_type,
              size_t inputs_memory,
              size_t outputs_memory,
              size_t weights_memory);
  /**
   * @brief Return the sum of inputs_memory, outputs_memory, and weights_memory
   * recorded in this CostMetrics.
   */
  size_t total_memory() const;

  /**
   * @brief Return the sum of memory recorded in this CostMetrics, but in MB,
   * instead of Bytes.
   */
  float total_memory_in_mb() const;

  /**
   * @brief Get the incremental difference between the total memory in
   * CostMetrics and sim->offset.
   * @details This is to easily compute the difference between sim->offset and
   * sum of all memory usage recorded in this CostMetrics.
   *
   * @param sim_offset Simulator->offset
   * @return size_t The incremental memory usage difference
   */
  size_t total_mem_diff_from(off_t sim_offset) const;

public:
  float forward_time;
  float backward_time;
  float sync_time;
  ///< Bytes of memory usage of different parts
  // Assume:
  // 1. all memory allocations use Simulator::allocate
  // 2. we call Simulator::free_all before measuring an operator
  // Therefore, the current memory usage of an operator is (size_t)sim->offset
  size_t inputs_memory;
  size_t outputs_memory;
  size_t weights_memory;
  ///< Memory usage of Op* considering parallelization over devices
  size_t op_total_mem;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::CostMetrics,
                 forward_time,
                 backward_time,
                 sync_time,
                 inputs_memory,
                 outputs_memory,
                 weights_memory,
                 op_total_mem);
MAKE_VISIT_HASHABLE(::FlexFlow::CostMetrics);

#endif
