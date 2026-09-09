#include "realm-execution/pcg_instance.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/optimizer_attrs.h"
#include "realm-execution/dependency_set.h"
#include "realm-execution/distributed_per_device_op_state_initialization.h"
#include "realm-execution/instance_allocation.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/redops/redop_id_t.h"
#include "realm-execution/tasks/impl/op_task.h"
#include "realm-execution/tensor_instance_backing.h"
#include "task-spec/dynamic_graph/copy_insertion.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_task_type.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/loss_insertion.h"
#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_mapped_pcg.h"
#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "task-spec/dynamic_graph/update_insertion.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/containers/try_at.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/optional.h"
#include <vector>

namespace FlexFlow {

PCGInstance::PCGInstance(
    RealmContext &ctx,
    std::vector<DynamicNodeInvocation> const &execution_order,
    TensorInstanceBacking const &tensor_instance_backing,
    PerDeviceOpStateBacking const &device_state_backing,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<Realm::RegionInstance> logit_grad_tensor)
    : ctx(ctx), execution_order(execution_order),
      tensor_instance_backing(tensor_instance_backing),
      device_state_backing(device_state_backing),
      optimizer_attrs(optimizer_attrs), logit_grad_tensor(logit_grad_tensor) {}

PCGInstance::~PCGInstance() {
  destroy_instances(this->tensor_instance_backing,
                    ctx.get_outstanding_events());
}

RealmContext &PCGInstance::get_realm_context() {
  return this->ctx;
}

std::vector<DynamicNodeInvocation> const &
    PCGInstance::get_execution_order() const {
  return this->execution_order;
}

TensorInstanceBacking const &PCGInstance::get_tensor_instance_backing() const {
  return this->tensor_instance_backing;
}

PerDeviceOpStateBacking const &PCGInstance::get_device_state_backing() const {
  return this->device_state_backing;
}

OptimizerAttrs const &PCGInstance::get_optimizer_attrs() const {
  return this->optimizer_attrs;
}

void PCGInstance::update_optimizer_attrs_for_next_iter() {
  this->optimizer_attrs =
      get_optimizer_attrs_for_next_iter(this->optimizer_attrs);
}

std::optional<Realm::RegionInstance>
    PCGInstance::get_loss_tensor_instance() const {
  return this->logit_grad_tensor;
}

PCGInstance create_pcg_instance(
    RealmContext &ctx,
    MappedParallelComputationGraph const &mpcg,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<ParallelLossConfig> const &loss,
    std::map<DynamicValueAttrs, DynamicTensorAccessor> const &input_tensors,
    ProfilingSettings const &profiling_settings,
    DistributedFfHandle const &device_handle,
    DeviceType device_type) {

  DynamicOpenDataflowGraph dg =
      make_dynamic_open_dataflow_graph_from_mapped_pcg(mpcg, device_type);
  dg = perform_pass_expansion(dg);

  std::map<DynamicValueAttrs, DynamicTensorAccessor> inputs = input_tensors;
  std::optional<DynamicValueAttrs> logit_grad_value;
  if (loss.has_value()) {
    ParallelLossConfig loss_config = assert_unwrap(loss);

    LossAttrs loss_attrs = loss_config.loss_attrs;
    GenericTensorAccessorR label_tensor = loss_config.label_tensor;
    parallel_tensor_guid_t logit_tensor = loss_config.logit_tensor;
    MappedOperatorTaskGroup loss_op_task_group = loss_config.loss_mapping;

    DynamicNodeMapping mapping = DynamicNodeMapping{
        /*op_task_group=*/loss_op_task_group,
        /*device_type=*/device_type,
    };

    auto [dg2, label_v, logit_grad_v] = perform_loss_insertion(
        dg, loss_attrs, dynamic_tensor_guid_t{logit_tensor}, mapping);
    dg = dg2;
    logit_grad_value = logit_grad_v;
    inputs.insert(std::pair{label_v, label_tensor});
  }

  dg = perform_update_insertion(dg, optimizer_attrs);
  dg = perform_copy_insertion(dg);
  dg = perform_shard_expansion(dg);

  TensorInstanceBacking tensor_instance_backing =
      perform_instance_allocation(dg, inputs, ctx);

  logit_grad_value =
      transform(logit_grad_value, [&](DynamicValueAttrs const &lgv) {
        for (DynamicNodeInvocation const &invocation : dg.invocations) {
          if (invocation.node_attrs.task_type != DynamicTaskType::LOSS) {
            continue;
          }
          for (auto const &[slot, value] : invocation.outputs) {
            if (slot.slot_name == TensorSlotName::LOGIT &&
                value.tensor_guid == lgv.tensor_guid &&
                value.role == lgv.role) {
              return value;
            }
          }
        }
        PANIC("couldn't find updated logit grad in the shard-expanded dynamic "
              "graph");
      });

  std::optional<Realm::RegionInstance> logit_grad_tensor =
      transform(logit_grad_value, [&](DynamicValueAttrs const &lgv) {
        return tensor_instance_backing.backing.at(lgv).first;
      });

  PerDeviceOpStateBacking device_state_backing =
      perform_distributed_per_device_op_state_initialization(
          ctx,
          dg,
          tensor_instance_backing,
          profiling_settings,
          device_handle,
          optimizer_attrs,
          ctx.get_outstanding_events());

  // Compute the topological ordering of the graph
  auto [kwarg_graph, node_map] =
      labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(dg);
  std::vector<Node> node_topo_order = get_topological_ordering(kwarg_graph);
  std::vector<DynamicNodeInvocation> invocation_topo_order = transform(
      node_topo_order, [&](Node node) { return node_map.at_l(node); });

  return PCGInstance{
      /*ctx=*/ctx,
      /*execution_order=*/invocation_topo_order,
      /*tensor_instance_backing=*/tensor_instance_backing,
      /*device_state_backing=*/device_state_backing,
      /*optimizer_attrs=*/optimizer_attrs,
      /*logit_grad_tensor=*/logit_grad_tensor,
  };
}

static Realm::Event
    issue_p2p_copy(RealmContext &ctx,
                   DynamicValueAttrs const &input,
                   DynamicValueAttrs const &output,
                   TensorInstanceBacking const &tensor_instance_backing,
                   Realm::Event precondition) {
  Realm::RegionInstance src_inst =
      tensor_instance_backing.backing.at(input).first;
  Realm::RegionInstance dst_inst =
      tensor_instance_backing.backing.at(output).first;
  return ctx.issue_copy(assert_unwrap(input.parallel_tensor_shape),
                        src_inst,
                        assert_unwrap(output.parallel_tensor_shape),
                        dst_inst,
                        Realm::ProfilingRequestSet{},
                        precondition);
}

static Realm::Event
    issue_p2p_reduction(RealmContext &ctx,
                        DynamicValueAttrs const &input,
                        DynamicValueAttrs const &output,
                        TensorInstanceBacking const &tensor_instance_backing,
                        redop_id_t redop_id,
                        bool is_fold,
                        bool exclusive,
                        Realm::Event precondition) {
  Realm::RegionInstance src_inst =
      tensor_instance_backing.backing.at(input).first;
  Realm::RegionInstance dst_inst =
      tensor_instance_backing.backing.at(output).first;
  return ctx.issue_reduction(assert_unwrap(input.parallel_tensor_shape),
                             src_inst,
                             assert_unwrap(output.parallel_tensor_shape),
                             dst_inst,
                             redop_id,
                             is_fold,
                             exclusive,
                             Realm::ProfilingRequestSet{},
                             precondition);
}

static Realm::Event issue_collective_broadcast(
    RealmContext &ctx,
    DynamicValueAttrs const &input,
    std::vector<DynamicValueAttrs> const &outputs,
    TensorInstanceBacking const &tensor_instance_backing,
    Realm::Event precondition) {
  // For now we just implement this as the naive set of N p2p copies.
  std::vector<Realm::Event> result =
      transform(outputs, [&](DynamicValueAttrs const &output) {
        return issue_p2p_copy(
            ctx, input, output, tensor_instance_backing, precondition);
      });
  return Realm::Event::merge_events(result);
}

static Realm::Event issue_collective_reduction(
    RealmContext &ctx,
    std::vector<DynamicValueAttrs> const &inputs,
    DynamicValueAttrs const &output,
    TensorInstanceBacking const &tensor_instance_backing,
    redop_id_t redop_id,
    Realm::Event precondition) {
  // For now we just implement this as a naive set of N p2p reductions. Because
  // we're launching them in parallel they cannot be exclusive (i.e., they need
  // to use per-element atomics to update the output tensor)
  std::vector<Realm::Event> result =
      transform(inputs, [&](DynamicValueAttrs const &input) {
        return issue_p2p_reduction(ctx,
                                   input,
                                   output,
                                   tensor_instance_backing,
                                   redop_id,
                                   /*is_fold*/ false,
                                   /*exclusive*/ false,
                                   precondition);
      });
  return Realm::Event::merge_events(result);
}

/**
 * \brief Spawn the Realm operations (tasks, copies, etc.) for a given \ref
 * DynamicNodeInvocation, given the specified dependencies, instances, etc. Note
 * that one \ref DynamicNodeInvocation may become multiple Realm operations
 * (e.g., a parallel operator may turn into multiple copies).
 */
static Realm::Event spawn_dynamic_node_invocation(
    RealmContext &ctx,
    DynamicNodeInvocation const &invocation,
    std::vector<Realm::Event> const &input_dependencies,
    std::vector<Realm::Event> const &output_dependencies,
    TensorInstanceBacking const &tensor_instance_backing,
    PerDeviceOpStateBacking const &device_state_backing,
    OptimizerAttrs const &optimizer_attrs,
    ProfilingSettings const &profiling_settings,
    DistributedFfHandle const &device_handle) {
  Realm::Event precondition = Realm::Event::merge_events(
      Realm::Event::merge_events(input_dependencies),
      Realm::Event::merge_events(output_dependencies));

  TensorInstanceBacking tensor_backing =
      subset_tensor_instance_backing_for_invocation(tensor_instance_backing,
                                                    invocation);

  auto spawn_task = [&]() {
    Realm::Processor target_proc = ctx.processor_from_global_device_id(
        get_only(assert_unwrap(invocation.node_attrs.device_ids)));
    return spawn_op_task(ctx,
                         target_proc,
                         invocation,
                         tensor_backing,
                         try_at(device_state_backing.backing, invocation),
                         profiling_settings,
                         device_handle.at(target_proc),
                         optimizer_attrs,
                         precondition);
  };

  auto issue_copy = [&]() {
    DynamicValueAttrs const &input = get_only(invocation.inputs).second;
    DynamicValueAttrs const &output = get_only(invocation.outputs).second;
    return issue_p2p_copy(
        ctx, input, output, tensor_instance_backing, precondition);
  };

  auto issue_replicate = [&]() {
    DynamicValueAttrs const &input = get_only(invocation.inputs).second;
    std::vector<DynamicValueAttrs> outputs =
        vector_of(values(invocation.outputs));
    return issue_collective_broadcast(
        ctx, input, outputs, tensor_instance_backing, precondition);
  };

  auto issue_reduction = [&]() {
    std::vector<DynamicValueAttrs> inputs =
        vector_of(values(invocation.inputs));
    DynamicValueAttrs const &output = get_only(invocation.outputs).second;
    redop_id_t redop_id = get_sum_redop_id_for_data_type(
        assert_unwrap(output.parallel_tensor_shape).data_type);
    return issue_collective_reduction(
        ctx, inputs, output, tensor_instance_backing, redop_id, precondition);
  };

  TrainingOperationAttrs op_attrs =
      assert_unwrap(invocation.node_attrs.op_attrs);
  return op_attrs.visit<Realm::Event>(overload{
      [&](PCGOperatorAttrs const &pcg_op_attrs) {
        return pcg_op_attrs.visit<Realm::Event>(overload{
            [&](InputAttrs const &) { return Realm::Event::NO_EVENT; },
            [&](WeightAttrs const &) { return Realm::Event::NO_EVENT; },
            [&](ReplicateAttrs const &) {
              DynamicTaskType task_type =
                  assert_unwrap(invocation.node_attrs.task_type);
              switch (task_type) {
                case DynamicTaskType::FWD:
                  return issue_replicate();
                case DynamicTaskType::BWD:
                  return issue_reduction();
                default:
                  PANIC("Unhandled replicate task type ", task_type);
              }
            },
            [&](auto const &) { return spawn_task(); },
        });
      },
      [&](LossAttrs const &) { return spawn_task(); },
      [&](CopyAttrs const &) { return issue_copy(); },
      [&](GradientReductionAttrs const &) { return issue_reduction(); },
  });
}

static std::map<dynamic_layer_guid_t, Realm::Event>
    execute_distributed_dynamic_node_invocation_set(
        RealmContext &ctx,
        std::vector<DynamicNodeInvocation> const &invocations,
        TensorInstanceBacking const &tensor_instance_backing,
        PerDeviceOpStateBacking const &device_state_backing,
        OptimizerAttrs const &optimizer_attrs,
        ProfilingSettings const &profiling_settings,
        DistributedFfHandle const &device_handle) {
  // For simplicity we'll track a dependency on all outstanding operations up to
  // this point. This will create an effective barrier between phases.
  DependencySet dependency_set{ctx.get_outstanding_events()};
  return map_from_pairs(
      transform(invocations, [&](DynamicNodeInvocation const &invocation) {
        std::vector<Realm::Event> input_dependencies =
            transform(vector_of(values(invocation.inputs)),
                      [&](DynamicValueAttrs const &value) {
                        return dependency_set.get_dependency_for_reader(value);
                      });
        std::vector<Realm::Event> output_dependencies =
            transform(vector_of(values(invocation.outputs)),
                      [&](DynamicValueAttrs const &value) {
                        return dependency_set.get_dependency_for_writer(value);
                      });

        Realm::Event result =
            spawn_dynamic_node_invocation(ctx,
                                          invocation,
                                          input_dependencies,
                                          output_dependencies,
                                          tensor_instance_backing,
                                          device_state_backing,
                                          optimizer_attrs,
                                          profiling_settings,
                                          device_handle);

        for (DynamicValueAttrs const &value : values(invocation.inputs)) {
          dependency_set.add_reader(value, result);
        }
        for (DynamicValueAttrs const &value : values(invocation.outputs)) {
          dependency_set.add_writer(value, result);
        }
        return std::pair{invocation.node_attrs.layer_guid, result};
      }));
}

std::map<dynamic_layer_guid_t, Realm::Event>
    perform_all_passes_for_pcg_instance(
        PCGInstance &pcg_instance,
        ProfilingSettings const &profiling_settings,
        DistributedFfHandle const &device_handle) {
  std::vector<DynamicNodeInvocation> execution_order =
      pcg_instance.get_execution_order();
  std::map<dynamic_layer_guid_t, Realm::Event> result =
      execute_distributed_dynamic_node_invocation_set(
          /*ctx=*/pcg_instance.get_realm_context(),
          /*invocations=*/execution_order,
          /*tensor_instance_backing=*/
          pcg_instance.get_tensor_instance_backing(),
          /*device_state_backing=*/pcg_instance.get_device_state_backing(),
          /*optimizer_attrs=*/pcg_instance.get_optimizer_attrs(),
          /*profiling_settings=*/profiling_settings,
          /*device_handle=*/device_handle);
  pcg_instance.update_optimizer_attrs_for_next_iter();
  return result;
}

std::map<dynamic_layer_guid_t, Realm::Event>
    perform_forward_pass_for_pcg_instance(
        PCGInstance &pcg_instance,
        ProfilingSettings const &profiling_settings,
        DistributedFfHandle const &device_handle) {
  std::vector<DynamicNodeInvocation> execution_order =
      filter(pcg_instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::FWD;
             });

  return execute_distributed_dynamic_node_invocation_set(
      /*ctx=*/pcg_instance.get_realm_context(),
      /*invocations=*/execution_order,
      /*tensor_instance_backing=*/pcg_instance.get_tensor_instance_backing(),
      /*device_state_backing=*/pcg_instance.get_device_state_backing(),
      /*optimizer_attrs=*/pcg_instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*device_handle=*/device_handle);
}

std::map<dynamic_layer_guid_t, Realm::Event>
    perform_backward_pass_for_pcg_instance(
        PCGInstance &pcg_instance,
        ProfilingSettings const &profiling_settings,
        DistributedFfHandle const &device_handle) {
  std::vector<DynamicNodeInvocation> execution_order =
      filter(pcg_instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::BWD;
             });

  return execute_distributed_dynamic_node_invocation_set(
      /*ctx=*/pcg_instance.get_realm_context(),
      /*invocations=*/execution_order,
      /*tensor_instance_backing=*/pcg_instance.get_tensor_instance_backing(),
      /*device_state_backing=*/pcg_instance.get_device_state_backing(),
      /*optimizer_attrs=*/pcg_instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*device_handle=*/device_handle);
}

std::map<dynamic_layer_guid_t, Realm::Event>
    perform_update_pass_for_pcg_instance(
        PCGInstance &pcg_instance,
        ProfilingSettings const &profiling_settings,
        DistributedFfHandle const &device_handle) {
  std::vector<DynamicNodeInvocation> execution_order =
      filter(pcg_instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::UPD;
             });

  std::map<dynamic_layer_guid_t, Realm::Event> result =
      execute_distributed_dynamic_node_invocation_set(
          /*ctx=*/pcg_instance.get_realm_context(),
          /*invocations=*/execution_order,
          /*tensor_instance_backing=*/
          pcg_instance.get_tensor_instance_backing(),
          /*device_state_backing=*/pcg_instance.get_device_state_backing(),
          /*optimizer_attrs=*/pcg_instance.get_optimizer_attrs(),
          /*profiling_settings=*/profiling_settings,
          /*device_handle=*/device_handle);
  pcg_instance.update_optimizer_attrs_for_next_iter();
  return result;
}

} // namespace FlexFlow
