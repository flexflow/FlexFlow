#include "tensor_args_format.h"
#include "utils/containers/flatmap.h"

namespace FlexFlow {

bool includes_tensor(ExecutableTensorSpec const &spec,
                     parallel_tensor_guid_t const &guid) {
  if (is_variadic(spec)) {
    return contains(get_variadic(spec), guid);
  } else {
    assert(is_nonvariadic(spec));
    return get_nonvariadic(spec) == guid;
  }
}

std::unordered_set<slot_id>
    get_tensor_slots(ExecutableTaskBinding const &binding,
                     parallel_tensor_guid_t const &guid) {
  std::unordered_set<slot_id> results;
  for (auto const &kv : binding.tensor_bindings) {
    slot_id slot = kv.first;
    ExecutableTensorSpec spec = kv.second;
    if (includes_tensor(spec, guid)) {
      results.insert(slot);
    }
  }
  return results;
}

Permissions get_tensor_permissions(TaskSignature const &sig,
                                   ExecutableTaskBinding const &binding,
                                   parallel_tensor_guid_t const &guid) {
  Permissions result = Permissions::NONE;
  for (slot_id slot : get_tensor_slots(binding, guid)) {
    result = join(result, sig.get_slot(slot)->perm);
  }
  return result;
}

std::vector<parallel_tensor_guid_t>
    as_vector(ExecutableTensorSpec const &spec) {
  if (is_variadic(spec)) {
    return get_variadic(spec);
  } else {
    assert(is_nonvariadic(spec));
    return {get_nonvariadic(spec)};
  }
}

TensorArgsFormat process_tensor_args(TaskSignature const &sig,
                                     ParallelComputationGraph const &pcg,
                                     ExecutableTaskBinding const &binding) {
  std::unordered_map<parallel_tensor_guid_t, Permissions> privs_map;
  bidict<parallel_tensor_guid_t, region_idx_t> region_idxs;
  std::unordered_map<parallel_tensor_guid_t, DataType> datatypes;
  int idx_ctr = 0;
  for (parallel_tensor_guid_t const &guid :
       unique(flatmap(values(binding.tensor_bindings), as_vector))) {
    for (slot_id slot : get_tensor_slots(binding, guid)) {
      privs_map[guid] = get_tensor_permissions(sig, binding, guid);
      region_idx_t idx = region_idx_t(idx_ctr++);
      region_idxs.equate(guid, idx);
      datatypes[guid] = pcg.at(guid).data_type;
    }
  }
  std::unordered_map<slot_id, parallel_tensor_guid_t>
      nonvariadic_slot_to_tensor;
  std::unordered_map<slot_id, std::vector<parallel_tensor_guid_t>>
      variadic_slot_to_tensor;
  for (slot_id slot : keys(binding.tensor_bindings)) {
    if (is_variadic(binding, slot)) {
      variadic_slot_to_tensor[slot] =
          get_variadic(binding.tensor_bindings.at(slot));
    } else {
      assert(is_nonvariadic(binding, slot));
      nonvariadic_slot_to_tensor[slot] =
          get_nonvariadic(binding.tensor_bindings.at(slot));
    }
  }

  return {region_idxs,
          privs_map,
          datatypes,
          nonvariadic_slot_to_tensor,
          variadic_slot_to_tensor};
}

} // namespace FlexFlow
