#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H

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

using ArgType = variant<TaskArgumentFormat, SimArg>;
using ArgVariadicType = variant<std::vector<TaskArgumentFormat>, std::vector<SimArg>>;

using PrivilegeType = variant<privilege_mode_to_accessor<Permissions::RW>,
                              privilege_mode_to_accessor<Permissions::RO>,
                              privilege_mode_to_accessor<Permissions::WO>>;
using PrivilegeVariadicType = variant<std::vector<privilege_mode_to_accessor<Permissions::RW>>,
                                    std::vector<privilege_mode_to_accessor<Permissions::RO>>,
                                    std::vector<privilege_mode_to_accessor<Permissions::WO>>>;

struct ITaskArgumentAccessor {
  virtual PrivilegeType get_tensor(slot_id slot, Permissions priv) const = 0;

  virtual PrivilegeVariadicType get_variadic_tensor(slot_id slot, Permissions priv) const = 0;

  virtual optional<ArgType> get_optional_argument(slot_id) const = 0;

  virtual ArgVariadicType get_variadic_argument(slot_id) const = 0;

  virtual PrivilegeType get_generic_accessor(region_idx_t const &idx, Permissions priv) const = 0;

  virtual PrivilegeType get_tensor_grad(slot_id slot, Permissions priv) const = 0;

  virtual PrivilegeVariadicType get_variadic_tensor_grad(slot_id slot, Permissions priv) const = 0;

  virtual size_t get_device_idx() const = 0;
};


template <typename T>
struct ArgumentAccessor {
  T operator()(LocalTaskArgumentAccessor& acc, slot_id slot) {
    return acc.get_argument<T>(slot);
  }

  T operator()(LegionTaskArgumentAccessor& acc, slot_id slot) {
    return acc.get_argument<T>(slot);
  }
};

template <typename T>
struct OptionalArgumentAccessor {
  T operator()(LocalTaskArgumentAccessor& acc, slot_id slot) {
    return acc.get_optional_argument<T>(slot);
  }

  T operator()(LegionTaskArgumentAccessor& acc, slot_id slot) {
    return acc.get_optional_argument<T>(slot);
  }
};

template <typename T>
struct VariadicArgumentAccessor {
  T operator()(LocalTaskArgumentAccessor& acc, slot_id slot) {
    return acc.get_variadic_argument<T>(slot);
  }

  T operator()(LegionTaskArgumentAccessor& acc, slot_id slot) {
    return acc.get_variadic_argument<T>(slot);
  }
};

struct LegionTaskArgumentAccessor : public ITaskArgumentAccessor {
public:
  template <typename T>
  T const &get_argument(slot_id slot) const;

  PrivilegeType get_tensor(slot_id slot, Permissions priv) const override;

  PrivilegeVariadicType get_variadic_tensor(slot_id slot, Permissions priv) const override;

  template <typename T>
  optional<T> get_optional_argument(slot_id) const;

  template <typename T>
  std::vector<T> get_variadic_argument(slot_id) const;

  PrivilegeType get_tensor_grad(slot_id slot, Permissions priv) const override;

  PrivilegeVariadicType get_variadic_tensor_grad(slot_id slot, Permissions priv) const override;

  size_t get_device_idx() const override;

  LegionTaskArgumentAccessor(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime)
      : task(task), regions(regions), ctx(ctx), runtime(runtime) {}

private:
  Legion::Task const *task;
  std::vector<Legion::PhysicalRegion> const &regions;
  Legion::Context ctx;
  Legion::Runtime *runtime;
  TaskArgumentsFormat const &args_fmt;
};

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {
public:
  template <typename T>
  T const &get_argument(slot_id slot) const;

  PrivilegeType get_tensor(slot_id slot, Permissions priv) const override;

  PrivilegeVariadicType get_variadic_tensor(slot_id slot, Permissions priv) const override;

  template <typename T>
  optional<T> get_optional_argument(slot_id) const;

  template <typename T>
  std::vector<T> get_variadic_argument(slot_id) const;

  PrivilegeType get_tensor_grad(slot_id slot, Permissions priv) const override;

  PrivilegeVariadicType get_variadic_tensor_grad(slot_id slot, Permissions priv) const override;

  size_t get_device_idx() const override;

  LocalTaskArgumentAccessor(
      std::shared_ptr<SimTaskBinding const> &sim_task_binding)
      : sim_task_binding(sim_task_binding), memory_usage(0) {
    local_allocator = Allocator::create<CudaAllocator>();
  }

  size_t get_memory_usage() const {
    return memory_usage;
  }

  void *allocate(size_t size);
  void deallocate(void *ptr);

private:
  std::shared_ptr<SimTaskBinding const> sim_task_binding;
  Allocator local_allocator;
  size_t memory_usage;
};

struct TaskArgumentAccessor {
  template <typename T>
  T const &get_argument(slot_id slot) const {
    return std::visit(ArgumentAccessor(), this->ptr, slot);
  }

  template <typename T>
  optional<T> const &get_optional_argument(slot_id slot) const {
    return std::visit(OptionalArgumentAccessor(), this->ptr, slot);
  }

  template <typename T>
  std::vector<T> const &get_variadic_argument(slot_id slot) const {
    return std::visit(VariadicArgumentAccessor(), this->ptr, slot);
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor(slot_id slot) const {
    return this->ptr->get_tensor(slot, PRIV);
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_tensor(slot_id slot) const {
    return this->ptr->get_variadic_tensor(slot, PRIV);
  }

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<ITaskArgumentAccessor, T>::value,
                              TaskArgumentAccessor>::type
      create(Args &&...args) {
    return TaskArgumentAccessor(
        std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  TaskArgumentAccessor(std::shared_ptr<ITaskArgumentAccessor const> &ptr)
      : ptr(ptr) {}
  std::shared_ptr<ITaskArgumentAccessor const> ptr;
};

} // namespace FlexFlow

#endif
