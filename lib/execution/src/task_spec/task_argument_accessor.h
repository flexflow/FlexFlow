#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H

#include "accessor.h"
#include "device_specific.h"
#include "realm_allocator.h"
#include "runtime/config.h"
#include "utils/exception.h"
#include "utils/stack_map.h"
#include "utils/strong_typedef.h"
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

struct TaskArgumentAccessor {
  TaskArgumentAccessor(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);

  Allocator get_allocator() const {
    return get_gpu_memory_allocator(this->task);
  }

  template <typename T>
  T const &get_argument(slot_id slot) const {
    NOT_IMPLEMENTED();
    // TaskArgumentFormat arg_fmt = this->args_fmt.args.at(slot);
    // std::type_index actual_type = arg_fmt.type;
    // std::type_index requested_type = {typeid(T)};

    // if (actual_type != requested_type) {
    //   throw mk_runtime_error(
    //       "Type mismatch in argument access (\"{}\" != \"{}\")",
    //       actual_type.name(),
    //       requested_type.name());
    // }

    // void *start_ptr = &((std::uint8_t *)this->task->args)[arg_fmt.start];
    // Legion::Deserializer dez(start_ptr, arg_fmt.start);

    // return ff_task_deserialize<T>(dez);
  }

  template <typename T>
  optional<T> get_optional_argument(slot_id) const {
    NOT_IMPLEMENTED();
  }

  template <typename T>
  std::vector<T> get_variadic_argument(slot_id) const {
    NOT_IMPLEMENTED();
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV>
      get_generic_accessor(region_idx_t const &idx) const {
    auto tensor_privs = get_permissions(this->args_fmt, idx);
    if (tensor_privs != PRIV) {
      throw mk_runtime_error(
          "Privilege mismatch while accessing tensor: {} != {}",
          tensor_privs,
          PRIV);
    }

    return helperGetGenericTensorAccessor<PRIV>(
        get_datatype(this->args_fmt, idx),
        regions[idx.value()],
        task->regions[idx.value()],
        FID_DATA,
        ctx,
        runtime);
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor(slot_id slot) const {
    auto argument_format =
        get<NonvariadicFormat>(this->args_fmt.region_idxs.at(slot));

    return this->get_generic_accessor<PRIV>(argument_format);
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor_grad(slot_id slot) const {
    NOT_IMPLEMENTED();
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_tensor(slot_id slot) const {
    std::vector<privilege_mode_to_accessor<PRIV>> result;

    auto argument_format =
        get<VariadicFormat>(this->args_fmt.region_idxs.at(slot));
    for (NonvariadicFormat const &argument : argument_format) {
      result.push_back(this->get_generic_accessor<PRIV>(argument));
    }

    return result;
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_tensor_grad(slot_id slot) const {
    NOT_IMPLEMENTED();
  }

  template <typename T>
  T *unwrap(DeviceSpecific<T> const &arg) const {
    return arg.get(this->get_device_idx());
  }

  template <typename T, typename... Args>
  DeviceSpecific<T> create_device_specific(Args &&...args) const {
    return DeviceSpecific<T>::create(this->get_device_idx(),
                                     std::forward<Args>(args)...);
  }

  size_t get_device_idx() const {
    NOT_IMPLEMENTED();
  }

private:
  Legion::Task const *task;
  std::vector<Legion::PhysicalRegion> const &regions;
  Legion::Context ctx;
  Legion::Runtime *runtime;
  TaskArgumentsFormat const &args_fmt;
};

} // namespace FlexFlow

#endif
