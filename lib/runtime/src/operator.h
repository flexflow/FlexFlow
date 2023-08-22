#ifndef _FLEXFLOW_RUNTIME_SRC_OPERATOR_H
#define _FLEXFLOW_RUNTIME_SRC_OPERATOR_H

#include "kernels/profiling.h"
#include "layer_id.h"
#include "op-attrs/operator_attrs.h"
#include "parallel_tensor.h"
#include "pcg/machine_view.h"
#include "profiling.h"
#include "runtime/config.h"
#include "tasks.h"
#include "utils/stack_string.h"
#include "utils/stack_vector.h"
#include "utils/strong_typedef.h"
#include <stdexcept>
#include <vector>

namespace FlexFlow {

/* class Simulator; */
/* class CostMetrics; */
/* class FFModel; */

/* void inner_measure_operator_cost(Simulator *sim, */
/*                                  std::function<void(ffStream_t)> const
 * &forward, */
/*                                  std::function<void(ffStream_t)> const
 * &backward, */
/*                                  CostMetrics &cost_metrics); */

// class Operator {
// protected:
//
// public:
//   Operator(
//      std::string const &name,
//      PCGOperatorAttrs const &attrs);
//
//   // Pure virtual functions that must be implemented
//   // virtual void init(FFModel const &) = 0;
//   // virtual void forward(FFModel const &) = 0;
//   // virtual void backward(FFModel const &) = 0;
//
//   // virtual bool measure_operator_cost(Simulator *sim,
//   //                                    MachineView const &mv,
//   //                                    CostMetrics &cost_metrics) const = 0;
//   // virtual bool estimate_sync_cost(Simulator *sim,
//   //                                 MachineView const &pc,
//   //                                 CostMetrics &cost_metrics) const;
//   // //
//   // // Other virtual functions that can be optionally overwritten
//   // virtual Legion::Domain get_input_tensor_shape(MachineView const &pc,
//   //                                               int input_idx,
//   //                                               int part_idx) const; //
//   deprecated
//   // virtual Legion::Domain get_output_tensor_shape(MachineView const &pc,
//   //                                                int output_idx,
//   //                                                int part_idx) const; //
//   deprecated
//   // virtual Legion::Domain get_weight_tensor_shape(MachineView const &pc,
//   //                                                int weight_idx,
//   //                                                int part_idx) const; //
//   deprecated
//   // virtual bool is_valid_parallel_config(FFModel const &ff,
//   //                                       MachineView const &pc) const;
//   //
//   // // Helper functions
//   // void prefetch(FFModel const &);
//   // void zero_grad(FFModel const &);
//   // ParallelTensor get_parameter(int index);
//   // virtual void map_output_tensors(FFModel &ff);
//   // virtual bool can_inplace_output();
//   // virtual bool has_inplace_output();
//   // virtual void do_inplace_output();
//   // virtual bool is_parallel_op() const;
//
// //   int get_dimension() const;
// // #ifdef FF_USE_NCCL
// //   static ncclUniqueId get_nccl_unique_id_task(
// //       Legion::Task const *task,
// //       std::vector<Legion::PhysicalRegion> const &regions,
// //       Legion::Context ctx,
// //       Legion::Runtime *runtime);
// //   static ncclComm_t
// //       init_nccl_comms_task(Legion::Task const *task,
// //                            std::vector<Legion::PhysicalRegion> const
// &regions,
// //                            Legion::Context ctx,
// //                            Legion::Runtime *runtime);
// // #endif
// // protected:
//   // void set_argumentmap_for_init(FFModel const &ff, Legion::ArgumentMap
//   &argmap);
//   // void set_argumentmap_for_forward(FFModel const &ff,
//   //                                  Legion::ArgumentMap &argmap) const;
//   // void set_argumentmap_for_backward(FFModel const &ff,
//   //                                   Legion::ArgumentMap &argmap);
//   // void set_opmeta_from_futuremap(FFModel const &ff,
//   //                                Legion::FutureMap const &fm);
//
//   // bool check_output_input_weight_same_parallel_is() const;
//
//   /* ParallelTensor const &get_parallel_tensor(TensorSpec const &) const; */
//   /* TensorSpec input_tensor(int idx) const; */
//   /* OpTaskBinding get_task_binding(OpTaskType) const; */
//   /* void set_argumentmap(OpTaskType, FFModel const &f, Legion::ArgumentMap);
//   */
//
//   /* virtual OpTaskBinding get_init_task_binding() const = 0; */
//   /* virtual TaskID get_init_task_id() const = 0; */
//   /* virtual OpTaskBinding get_fwd_task_binding() const = 0; */
//   /* virtual TaskID get_fwd_task_id() const = 0; */
//   /* virtual OpTaskBinding get_bwd_task_binding() const = 0; */
//   /* virtual TaskID get_bwd_task_id() const = 0; */
// public:
//   stack_string<MAX_OPNAME> name;
//   /* Legion::IndexSpace parallel_is; */
//   PCGOperatorAttrs attrs;
// };

struct Operator : public use_visitable_cmp<Operator> {
public:
  Operator() = delete;
  Operator(std::string const &name, PCGOperatorAttrs const &attrs);

  operator PCGOperatorAttrs() const;

public:
  stack_string<MAX_OPNAME> name;
  PCGOperatorAttrs attrs;
};

static_assert(std::is_copy_constructible<Operator>::value, "");

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::Operator, name, attrs);
MAKE_VISIT_HASHABLE(::FlexFlow::Operator);

#endif
