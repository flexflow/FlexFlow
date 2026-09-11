#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_PCG_INSTANCE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_PCG_INSTANCE_H

#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/distributed_ff_handle.h"
#include "realm-execution/parallel_loss_config.dtg.h"
#include "realm-execution/per_device_op_state_backing.dtg.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/global_device_id_t.dtg.h"
#include "utils/units/milliseconds_t.h"
#include <optional>

namespace FlexFlow {

/**
 * \brief The main public interface for the Realm backend.
 * Takes a \ref MappedParallelComputationGraph and lowers it through
 * \ref DynamicOpenDataflowGraph to get the fully-specified execution order of
 * tasks to be issued. (Note: this is a parallel execution so execution order
 * may not match the order in which operations are issued.) Also tracks the
 * allocation of realm instances for tensors through its \ref
 * TensorInstanceBacking.
 *
 * \note \ref PCGInstance is primarily just a container for the various structs
 * held inside it. The actual initialization and training iteration
 * functionality is held in \ref create_pcg_instance and \ref
 * perform_update_pass_for_pcg_instance, respectively.
 *
 */
struct PCGInstance {
public:
  PCGInstance() = delete;
  PCGInstance(PCGInstance const &) = delete;
  PCGInstance(PCGInstance &&) = delete;

  explicit PCGInstance(
      RealmContext &ctx,
      std::vector<DynamicNodeInvocation> const &execution_order,
      TensorInstanceBacking const &tensor_instance_backing,
      PerDeviceOpStateBacking const &device_state_backing,
      OptimizerAttrs const &optimizer_attrs,
      std::optional<Realm::RegionInstance> logit_grad_tensor);

  ~PCGInstance();

  void update_optimizer_attrs_for_next_iter();

  /** \name Getters **/
  ///\{
  RealmContext &get_realm_context();
  std::vector<DynamicNodeInvocation> const &get_execution_order() const;
  TensorInstanceBacking const &get_tensor_instance_backing() const;
  PerDeviceOpStateBacking const &get_device_state_backing() const;
  OptimizerAttrs const &get_optimizer_attrs() const;
  std::optional<Realm::RegionInstance> get_loss_tensor_instance() const;
  ///\}

private:
  RealmContext &ctx;
  std::vector<DynamicNodeInvocation> execution_order;
  TensorInstanceBacking tensor_instance_backing;
  PerDeviceOpStateBacking device_state_backing;
  OptimizerAttrs optimizer_attrs;
  std::optional<Realm::RegionInstance> logit_grad_tensor;
};

/**
 * \brief Creates a \ref PCGInstance. Should generally be used instead of \ref
 * PCGInstance::PCGInstance.
 *
 * \relates PCGInstance
 */
PCGInstance create_pcg_instance(
    RealmContext &ctx,
    MappedParallelComputationGraph const &mpcg,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<ParallelLossConfig> const &loss,
    std::map<DynamicValueAttrs, DynamicTensorAccessor> const &input_tensors,
    DistributedFfHandle const &ff_handle,
    DeviceType device_type);

/**
 * \brief Dispatch a training iteration for a \ref PCGInstance.
 *
 * To dispatch just a piece of a training iteration, see the following
 * functions:
 * - \ref perform_forward_pass_for_pcg_instance
 * - \ref perform_backward_pass_for_pcg_instance
 * - \ref perform_update_pass_for_pcg_instance
 *
 * \relates PCGInstance
 */
std::map<dynamic_layer_guid_t, Realm::Event>
    perform_all_passes_for_pcg_instance(PCGInstance &pcg_instance,
                                        DistributedFfHandle const &ff_handle);

std::map<dynamic_layer_guid_t, Realm::Event>
    perform_forward_pass_for_pcg_instance(PCGInstance &pcg_instance,
                                          DistributedFfHandle const &ff_handle);

std::map<dynamic_layer_guid_t, Realm::Event>
    perform_backward_pass_for_pcg_instance(
        PCGInstance &pcg_instance, DistributedFfHandle const &ff_handle);

std::map<dynamic_layer_guid_t, Realm::Event>
    perform_update_pass_for_pcg_instance(PCGInstance &pcg_instance,
                                         DistributedFfHandle const &ff_handle);

} // namespace FlexFlow

#endif
