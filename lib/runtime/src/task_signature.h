#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SIGNATURE_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SIGNATURE_H

#include "utils/strong_typedef.h"
#include "legion.h"
#include "serialization.h"
#include "utils/type_index.h"
#include "tasks.h"
#include <unordered_map>
#include "permissions.h"

namespace FlexFlow {

enum class SlotType {
  TENSOR,
  VARIADIC
};

struct slot_id : strong_typedef<slot_id, int> {
  using strong_typedef::strong_typedef;

  slot_id(int);
};

}

MAKE_TYPEDEF_HASHABLE(::FlexFlow::slot_id);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::slot_id, "slot_id");

namespace FlexFlow {

struct ParallelTensorSlotSpec {
public:
  ParallelTensorSlotSpec() = delete;
  ParallelTensorSlotSpec(SlotType, Permissions perm);

public:
  SlotType slot_type;
  Permissions perm;
};

struct TaskSignature {
  TaskSignature() = default;

  void add_slot(slot_id, ParallelTensorSlotSpec const &);

  template <typename T>
  void add_arg_slot(slot_id name) {
    static_assert(is_serializable<T>, "Argument type must be serializable");

    this->task_arg_types.insert({ name, type_index<T>()});
  }

  template <typename T>
  void add_return_value();

  template <typename T>
  void add_variadic_arg_slot(slot_id name);

  optional<ParallelTensorSlotSpec> get_slot(slot_id) const;
  optional<std::type_index> get_arg_slot(slot_id) const;
  optional<std::type_index> get_return_type() const;

  /* template <typename T, typename F> */
  /* void add_index_arg_slot(slot_id name, F const &idx_to_arg) { */
  /*   static_assert(is_serializable<T>, "Argument type must be serializable"); */

  /*   this->task_arg_types.insert({ name, { typeid(T), ArgSlotType::INDEX }}); */
  /* } */

  bool operator==(TaskSignature const &) const;
  bool operator!=(TaskSignature const &) const;
private:
  std::unordered_map<slot_id, std::type_index> task_arg_types;
  std::unordered_map<slot_id, ParallelTensorSlotSpec> tensor_slots;
  optional<std::type_index> return_type;
};

TaskSignature get_signature(task_id_t);
std::string get_name(task_id_t);

template <typename F>
void register_task(task_id_t, std::string const &name, TaskSignature const &, F const &func);

template <typename F>
void register_task(task_id_t, std::string const &name, TaskSignature const &, F const &func, F const &cpu_func); 

}

#endif
