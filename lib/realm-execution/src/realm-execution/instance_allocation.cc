#include "realm-execution/instance_allocation.h"
#include "local-execution/tensor_allocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/make.h"
#include "utils/containers/map_values.h"
#include "utils/containers/set_of.h"
#include "utils/containers/values.h"
#include "utils/not_implemented.h"
#include "utils/optional.h"

namespace FlexFlow {

std::pair<Realm::RegionInstance, Realm::Event>
    perform_instance_allocation_for_value(global_device_id_t const &device_id,
                                          DynamicValueAttrs const &value,
                                          RealmContext &ctx) {
  ASSERT(value.accessor == std::nullopt);

  TensorShape shape = get_piece_shape(value.parallel_tensor_shape.value());

  Realm::Processor proc = ctx.processor_from_global_device_id(device_id);
  Realm::Memory memory = ctx.get_nearest_memory(proc);
  return ctx.create_instance(memory, shape, Realm::ProfilingRequestSet());
}

TensorInstanceBacking perform_instance_allocation(
    DynamicOpenDataflowGraph const &g,
    std::map<DynamicValueAttrs, DynamicTensorAccessor> const &preallocated,
    RealmContext &ctx) {
  ASSERT(no_tensors_are_allocated(g));
  ASSERT(tensors_are_ready_for_allocation(g));
  for (DynamicValueAttrs const &v : keys(preallocated)) {
    ASSERT(v.accessor == std::nullopt);
  }

  TensorInstanceBacking result = make_empty_tensor_instance_backing();
  auto allocate = [&](DynamicValueAttrs const &v) {
    if (contains_key(preallocated, v)) {
      // FIXME: Attach external instance to existing allocation and use that
      NOT_IMPLEMENTED();
    } else {
      if (!contains_key(result.backing, v)) {
        global_device_id_t device_id =
            assert_unwrap(v.mapping).raw.at_l(assert_unwrap(v.shard_coord));
        result.backing.insert(std::pair{
            v, perform_instance_allocation_for_value(device_id, v, ctx)});
      }
      return result.backing.at(v);
    }
  };

  for (DynamicNodeInvocation const &invocation : g.invocations) {
    for (DynamicValueAttrs const &input : values(invocation.inputs)) {
      allocate(input);
    }
    for (DynamicValueAttrs const &output : values(invocation.outputs)) {
      allocate(output);
    }
  }

  return result;
}

void destroy_instances(TensorInstanceBacking const &instances,
                       Realm::Event precondition) {
  for (auto const &[instance, ready] : values(instances.backing)) {
    instance.destroy(Realm::Event::merge_events(precondition, ready));
  }
}

} // namespace FlexFlow
