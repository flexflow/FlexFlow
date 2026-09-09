#include "realm-execution/realm_context.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_handle_t.h"
#include "op-attrs/datatype.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.dtg.h"
#include "realm-execution/address_space.h"
#include "realm-execution/fmt/realm_processor.h"
#include "realm-execution/fmt/realm_processor_kind.h"
#include "realm-execution/processor_kind.h"
#include "realm-execution/processor_query.h"
#include "realm-execution/realm_allocator.h"
#include "realm-execution/redops/redop_id_t.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "realm-execution/tasks/task_id_t.h"
#include "task-spec/global_device_id_t.h"
#include "utils/bidict/algorithms/bidict_from_enumerating.h"
#include "utils/bidict/algorithms/bidict_transform_values.h"
#include "utils/bidict/algorithms/merge_disjoint_bidicts.h"
#include "utils/containers/are_all_same.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/group_by.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/one_to_many/one_to_many.h"
#include "utils/optional.h"
#include "utils/positive_int/positive_int.h"

namespace FlexFlow {

bidict<Realm::Processor, local_device_id_t> build_local_machine_topology(
    std::set<Realm::Processor> const &local_procs) {
  {
    bool procs_are_local = are_all_same(
        transform(local_procs, [&](Realm::Processor p) -> Realm::AddressSpace {
          return p.address_space();
        }));
    ASSERT(procs_are_local);
  }

  OneToMany<Realm::Processor::Kind, Realm::Processor> by_proc_kind =
      group_by(local_procs, [](Realm::Processor p) -> Realm::Processor::Kind {
        return p.kind();
      });

  auto local_machine_topology_for_proc_kind = [&](Realm::Processor::Kind k)
      -> bidict<Realm::Processor, local_device_id_t> {
    if (!contains(by_proc_kind.left_values(), k)) {
      return {};
    }

    bidict<Realm::Processor, nonnegative_int> enumerated =
        bidict_from_enumerating(
            set_of(by_proc_kind.at_l(k).unwrap_as_unordered_set()))
            .reversed();

    bidict<Realm::Processor, local_device_id_t> result =
        bidict_transform_values(
            enumerated, [&](nonnegative_int idx) -> local_device_id_t {
              return local_device_id_t{
                  /*idx=*/device_in_node_idx_t{idx},
                  /*device_type=*/device_type_from_processor_kind(k),
              };
            });

    return result;
  };

  return binary_merge_disjoint_bidicts(
      local_machine_topology_for_proc_kind(Realm::Processor::Kind::LOC_PROC),
      local_machine_topology_for_proc_kind(Realm::Processor::Kind::TOC_PROC));
}

static bidict<Realm::Processor, global_device_id_t>
    build_global_machine_topology(
        std::set<Realm::Processor> const &global_procs) {
  OneToMany<node_idx_t, Realm::Processor> by_node_idx =
      group_by(global_procs, [](Realm::Processor p) -> node_idx_t {
        return node_idx_from_realm_address_space(p.address_space());
      });

  auto build_global_machine_topology_for_node = [&](node_idx_t const &node_idx)
      -> bidict<Realm::Processor, global_device_id_t> {
    std::set<Realm::Processor> procs_for_node =
        set_of(by_node_idx.at_l(node_idx).unwrap_as_unordered_set());

    bidict<Realm::Processor, local_device_id_t> local_topology_for_node =
        build_local_machine_topology(procs_for_node);

    return bidict_transform_values(
        local_topology_for_node,
        [&](local_device_id_t const &local_device_id) -> global_device_id_t {
          return global_device_id_from_local(local_device_id, node_idx);
        });
  };

  return merge_disjoint_bidicts(
      transform(set_of(by_node_idx.left_values()),
                build_global_machine_topology_for_node));
}

static bidict<Realm::Processor, local_device_id_t>
    discover_local_machine_topology(Realm::Processor local_processor) {
  Realm::Machine::ProcessorQuery pq(Realm::Machine::get_machine());
  pq.same_address_space_as(local_processor);

  return build_local_machine_topology(processor_set_from_query(pq));
}

static bidict<Realm::Processor, global_device_id_t>
    discover_global_machine_topology() {
  Realm::Machine::ProcessorQuery pq(Realm::Machine::get_machine());

  return build_global_machine_topology(processor_set_from_query(pq));
}

RealmContext::RealmContext(Realm::Processor processor)
    : processor(processor),
      allocator(get_realm_allocator(
          processor, RealmContext::get_nearest_memory(processor))),
      local_machine_topology(discover_local_machine_topology(processor)) {}

RealmContext::~RealmContext() {
  if (!this->outstanding_events.empty()) {
    Realm::Event outstanding = this->merge_outstanding_events();
    outstanding.wait();
  }
}

Realm::Processor RealmContext::processor_from_global_device_id(
    global_device_id_t const &global_device_id) {

  return this->get_global_machine_topology().at_r(global_device_id);
}

global_device_id_t
    RealmContext::global_device_id_from_processor(Realm::Processor processor) {

  return this->get_global_machine_topology().at_l(processor);
}

Realm::Processor RealmContext::processor_from_local_device_id(
    local_device_id_t const &local_device_id) const {

  return this->local_machine_topology.at_r(local_device_id);
}

local_device_id_t RealmContext::local_device_id_from_processor(
    Realm::Processor processor) const {

  return this->local_machine_topology.at_l(processor);
}

Realm::Memory RealmContext::get_nearest_memory(Realm::Processor proc) {
  if (!proc.exists()) {
    return Realm::Memory::NO_MEMORY;
  }

  Realm::Machine::MemoryQuery mq(Realm::Machine::get_machine());
  mq.best_affinity_to(proc);
  ASSERT(mq.count() > 0);

  // Among the best memories, pick the one with the largest capacity
  Realm::Memory result = Realm::Memory::NO_MEMORY;
  for (Realm::Machine::MemoryQuery::iterator it = mq.begin(); it != mq.end();
       ++it) {
    if (!result.exists() || it->capacity() > result.capacity()) {
      result = *it;
    }
  }

  ASSERT(result.exists());
  return result;
}

Realm::Processor RealmContext::get_current_processor() const {
  return this->processor;
}

Allocator &RealmContext::get_current_device_allocator() {
  return this->allocator;
}

global_device_id_t RealmContext::get_current_global_device_id() const {
  Realm::Processor proc = this->get_current_processor();

  return global_device_id_from_local(
      this->local_device_id_from_processor(proc),
      node_idx_from_realm_address_space(proc.address_space()));
}

Realm::Event
    RealmContext::spawn_task(Realm::Processor proc,
                             task_id_t task_id,
                             void const *args,
                             size_t arglen,
                             Realm::ProfilingRequestSet const &requests,
                             Realm::Event wait_on,
                             int priority) {
  Realm::Event result = proc.spawn(get_realm_task_id_for_task_id(task_id),
                                   args,
                                   arglen,
                                   requests,
                                   wait_on,
                                   priority);
  this->outstanding_events.push_back(result);
  return result;
}

Realm::Event RealmContext::collective_spawn_task(Realm::Processor target_proc,
                                                 task_id_t task_id,
                                                 void const *args,
                                                 size_t arglen,
                                                 Realm::Event wait_on,
                                                 int priority) {
  Realm::Event result =
      this->runtime.collective_spawn(target_proc,
                                     get_realm_task_id_for_task_id(task_id),
                                     args,
                                     arglen,
                                     wait_on,
                                     priority);
  this->outstanding_events.push_back(result);
  return result;
}

template <int N, typename T = int>
static Realm::Rect<N, T> rect_from_dims(TensorDims const &dims) {
  std::vector<int> values{dims.ff_ordered.begin(), dims.ff_ordered.end()};
  ASSERT(values.size() == N);
  return Realm::Rect<N, T>{Realm::Point<N, T>::ZEROES(),
                           Realm::Point<N, T>{values.data()} -
                               Realm::Point<N, T>::ONES()};
}

template <int N, typename T = int>
static Realm::IndexSpace<N, T> ispace_from_dims(TensorDims const &dims) {
  Realm::Rect<N, T> rect = rect_from_dims<N, T>(dims);
  return Realm::IndexSpace<N, T>{rect};
}

[[nodiscard]] static Realm::Event
    issue_copy_for_field(TensorDims const &dims,
                         Realm::CopySrcDstField const &src_field,
                         Realm::CopySrcDstField const &dst_field,
                         Realm::ProfilingRequestSet const &requests,
                         Realm::Event wait_on,
                         int priority) {
  switch (dims.ff_ordered.num_dims()) {
#if REALM_MAX_DIM >= 1
    case 1:
      return ispace_from_dims<1>(dims).copy(
          {src_field}, {dst_field}, requests, wait_on, priority);
#endif
#if REALM_MAX_DIM >= 2
    case 2:
      return ispace_from_dims<2>(dims).copy(
          {src_field}, {dst_field}, requests, wait_on, priority);
#endif
#if REALM_MAX_DIM >= 3
    case 3:
      return ispace_from_dims<3>(dims).copy(
          {src_field}, {dst_field}, requests, wait_on, priority);
#endif
#if REALM_MAX_DIM >= 4
    case 4:
      return ispace_from_dims<4>(dims).copy(
          {src_field}, {dst_field}, requests, wait_on, priority);
#endif
#if REALM_MAX_DIM >= 5
    case 5:
      return ispace_from_dims<5>(dims).copy(
          {src_field}, {dst_field}, requests, wait_on, priority);
#endif
    default:
      PANIC("TensorShape dims greater than REALM_MAX_DIM: {}",
            dims.ff_ordered.num_dims());
      break;
  }
}

Realm::Event
    RealmContext::issue_copy(ParallelTensorShape const &src_shape,
                             Realm::RegionInstance src_inst,
                             ParallelTensorShape const &dst_shape,
                             Realm::RegionInstance dst_inst,
                             Realm::ProfilingRequestSet const &requests,
                             Realm::Event wait_on,
                             int priority) {
  TensorShape src_piece_shape = get_piece_shape(src_shape);
  TensorShape dst_piece_shape = get_piece_shape(dst_shape);
  ASSERT(src_piece_shape == dst_piece_shape); // For now, assume they match

  Realm::CopySrcDstField src_field;
  src_field.set_field(
      /*inst=*/src_inst,
      /*field_id=*/0,
      /*size=*/
      static_cast<size_t>(
          size_of_datatype(src_piece_shape.data_type).int_from_positive_int()),
      /*subfield_offset=*/0);
  Realm::CopySrcDstField dst_field;
  dst_field.set_field(
      /*inst=*/dst_inst,
      /*field_id=*/0,
      /*size=*/
      static_cast<size_t>(
          size_of_datatype(src_piece_shape.data_type).int_from_positive_int()),
      /*subfield_offset=*/0);

  Realm::Event result = issue_copy_for_field(
      src_piece_shape.dims, src_field, dst_field, requests, wait_on, priority);
  this->outstanding_events.push_back(result);
  return result;
}

Realm::Event
    RealmContext::issue_reduction(ParallelTensorShape const &src_shape,
                                  Realm::RegionInstance src_inst,
                                  ParallelTensorShape const &dst_shape,
                                  Realm::RegionInstance dst_inst,
                                  redop_id_t redop_id,
                                  bool is_fold,
                                  bool exclusive,
                                  Realm::ProfilingRequestSet const &requests,
                                  Realm::Event wait_on,
                                  int priority) {
  TensorShape src_piece_shape = get_piece_shape(src_shape);
  TensorShape dst_piece_shape = get_piece_shape(dst_shape);
  ASSERT(src_piece_shape == dst_piece_shape); // For now, assume they match

  Realm::CopySrcDstField src_field;
  src_field.set_field(
      /*inst=*/src_inst,
      /*field_id=*/0,
      /*size=*/
      static_cast<size_t>(
          size_of_datatype(src_piece_shape.data_type).int_from_positive_int()),
      /*subfield_offset=*/0);
  Realm::CopySrcDstField dst_field;
  dst_field.set_field(
      /*inst=*/dst_inst,
      /*field_id=*/0,
      /*size=*/
      static_cast<size_t>(
          size_of_datatype(src_piece_shape.data_type).int_from_positive_int()),
      /*subfield_offset=*/0);
  dst_field.set_redop(
      get_realm_reduction_op_id_for_redop_id(redop_id), is_fold, exclusive);

  Realm::Event result = issue_copy_for_field(
      src_piece_shape.dims, src_field, dst_field, requests, wait_on, priority);
  this->outstanding_events.push_back(result);
  return result;
}

std::pair<Realm::RegionInstance, Realm::Event>
    RealmContext::create_instance(Realm::Memory memory,
                                  TensorShape const &shape,
                                  Realm::ProfilingRequestSet const &prs,
                                  Realm::Event wait_on) {
  std::vector<size_t> field_sizes{static_cast<size_t>(
      size_of_datatype(shape.data_type).int_from_positive_int())};
  Realm::RegionInstance inst;
  Realm::Event ready;
  switch (shape.dims.ff_ordered.num_dims()) {
#if REALM_MAX_DIM >= 1
    case 1:
      ready =
          Realm::RegionInstance::create_instance(inst,
                                                 memory,
                                                 rect_from_dims<1>(shape.dims),
                                                 field_sizes,
                                                 0 /*SOA*/,
                                                 prs,
                                                 wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 2
    case 2:
      ready =
          Realm::RegionInstance::create_instance(inst,
                                                 memory,
                                                 rect_from_dims<2>(shape.dims),
                                                 field_sizes,
                                                 0 /*SOA*/,
                                                 prs,
                                                 wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 3
    case 3:
      ready =
          Realm::RegionInstance::create_instance(inst,
                                                 memory,
                                                 rect_from_dims<3>(shape.dims),
                                                 field_sizes,
                                                 0 /*SOA*/,
                                                 prs,
                                                 wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 4
    case 4:
      ready =
          Realm::RegionInstance::create_instance(inst,
                                                 memory,
                                                 rect_from_dims<4>(shape.dims),
                                                 field_sizes,
                                                 0 /*SOA*/,
                                                 prs,
                                                 wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 5
    case 5:
      ready =
          Realm::RegionInstance::create_instance(inst,
                                                 memory,
                                                 rect_from_dims<5>(shape.dims),
                                                 field_sizes,
                                                 0 /*SOA*/,
                                                 prs,
                                                 wait_on);
      break;
#endif
    default:
      PANIC("TensorShape dims greater than REALM_MAX_DIM",
            fmt::to_string(shape.dims.ff_ordered.num_dims()));
      break;
  }
  this->outstanding_events.push_back(ready);
  return std::pair{inst, ready};
}

Realm::Event RealmContext::get_outstanding_events() {
  Realm::Event result = this->merge_outstanding_events();
  this->outstanding_events.push_back(result);
  return result;
}

Realm::Event RealmContext::merge_outstanding_events() {
  Realm::Event result = Realm::Event::merge_events(this->outstanding_events);
  this->outstanding_events.clear();
  return result;
}

bidict<Realm::Processor, global_device_id_t> const &
    RealmContext::get_global_machine_topology() {
  if (!this->cached_global_machine_topology.has_value()) {
    this->cached_global_machine_topology = discover_global_machine_topology();
  }

  return assert_unwrap(this->cached_global_machine_topology);
}

Realm::Runtime RealmContext::get_runtime() {
  return this->runtime;
}

} // namespace FlexFlow
