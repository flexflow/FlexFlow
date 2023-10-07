#ifndef _FLEXFLOW_RUNTIME_SRC_ITASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_RUNTIME_SRC_ITASK_ARGUMENT_ACCESSOR_H

#include "accessor.h"
#include "sim_environment.h"
#include "device_specific.h"
#include "realm_allocator.h"
#include "kernels/allocation.h"
#include "pcg/parallel_tensor_guid_t.h"
#include "runtime/config.h"
#include "utils/exception.h"
#include "utils/stack_map.h"
#include "utils/strong_typedef.h"
#include "utils/type_index.h"
#include <vector>

namespace FlexFlow {

template <Permissions PRIV>
using privilege_mode_to_accessor =
    typename privilege_mode_to_accessor_t<PRIV>::type;

struct region_idx_t : strong_typedef<region_idx_t, int> {
  using strong_typedef::strong_typedef;
};

FF_TYPEDEF_HASHABLE(region_idx_t);
FF_TYPEDEF_PRINTABLE(region_idx_t, "region_idx");

using NonvariadicFormat = region_idx_t;
using VariadicFormat = std::vector<NonvariadicFormat>;

using TensorArgumentFormat = variant<NonvariadicFormat, VariadicFormat>;

bool is_variadic(TensorArgumentFormat const &);
VariadicFormat get_variadic_format(TensorArgumentFormat const &);
NonvariadicFormat get_nonvariadic_format(TensorArgumentFormat const &);

struct TaskArgumentFormat {
  std::type_index type;
  size_t start;
  req<size_t> end;
};
FF_VISITABLE_STRUCT(TaskArgumentFormat, type, start, end);

struct FutureArgumentFormat {
  std::type_index type;
  req<size_t> future_idx;
};
FF_VISITABLE_STRUCT(FutureArgumentFormat, type, future_idx);

struct TaskArgumentsFormat {
  TaskArgumentsFormat() = default;

  stack_map<slot_id, TensorArgumentFormat, MAX_NUM_TASK_REGIONS> region_idxs;
  stack_map<slot_id, TaskArgumentFormat, MAX_NUM_TASK_ARGUMENTS> args;
  stack_map<slot_id, FutureArgumentFormat, MAX_NUM_TASK_ARGUMENTS> futures;
  stack_map<region_idx_t, Legion::PrivilegeMode, MAX_NUM_TASK_REGIONS> regions;
  stack_map<region_idx_t, DataType, MAX_NUM_TASK_REGIONS> data_types;

  void insert(std::pair<slot_id, TaskArgumentFormat> const &);
  void insert(std::pair<slot_id, FutureArgumentFormat> const &);

  void insert(region_idx_t, Legion::PrivilegeMode, DataType);
  void insert(slot_id, region_idx_t);
  void insert(slot_id, std::vector<region_idx_t> const &);
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(
    TaskArgumentsFormat, region_idxs, args, futures, regions, data_types);

Legion::PrivilegeMode get_privileges(TaskArgumentsFormat const &,
                                     region_idx_t const &);
Legion::PrivilegeMode get_privileges(TaskArgumentsFormat const &,
                                     parallel_tensor_guid_t const &);
Permissions get_permissions(TaskArgumentsFormat const &, region_idx_t const &);
Permissions get_permissions(TaskArgumentsFormat const &,
                            parallel_tensor_guid_t const &);
region_idx_t get_region_idx(TaskArgumentsFormat const &,
                            parallel_tensor_guid_t const &);
DataType get_datatype(TaskArgumentsFormat const &, region_idx_t const &);

using PrivilegeType = variant<privilege_mode_to_accessor<Permissions::RW>,
                              privilege_mode_to_accessor<Permissions::RO>,
                              privilege_mode_to_accessor<Permissions::WO>>;
using PrivilegeVariadicType = variant<std::vector<privilege_mode_to_accessor<Permissions::RW>>,
                                    std::vector<privilege_mode_to_accessor<Permissions::RO>>,
                                    std::vector<privilege_mode_to_accessor<Permissions::WO>>>;

struct ITaskArgumentAccessor {
  ITaskArgumentAccessor& operator=(const ITaskArgumentAccessor&) = delete;
  virtual ~ITaskArgumentAccessor() {};

  virtual PrivilegeType get_tensor(slot_id slot, Permissions priv) const = 0;

  virtual PrivilegeVariadicType get_variadic_tensor(slot_id slot, Permissions priv) const = 0;

  virtual PrivilegeType get_generic_accessor(region_idx_t const &idx, Permissions priv) const = 0;

  virtual PrivilegeType get_tensor_grad(slot_id slot, Permissions priv) const = 0;

  virtual PrivilegeVariadicType get_variadic_tensor_grad(slot_id slot, Permissions priv) const = 0;

  virtual size_t get_device_idx() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ITaskArgumentAccessor);

} // namespace FlexFlow

#endif