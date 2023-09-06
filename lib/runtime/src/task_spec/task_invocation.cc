#include "task_invocation.h"
#include "task_signature.h"
#include "legion.h"

using namespace Legion;

namespace FlexFlow {

void TaskBinding::insert_arg_spec(slot_id name, ArgSpec const &arg_spec) {
  assert(!contains_key(this->arg_bindings, name));
  this->arg_bindings.insert({name, arg_spec});
}

TaskSignature get_signature(task_id_t task_id) {
  if (TaskSignature::task_sig_map.count(task_id)) {
    return TaskSignature::task_sig_map.at(task_id);
  } else {
    throw mk_runtime_error("Unknown task id {}. Please report this as an issue.", task_id);
  }
}

template <typename F>
void register_task(task_id_t task_id,
                   std::string const &name,
                   F const &func,
                   optional<F const &> cpu_func = nullopt) {
  // gpu task
  {
    TaskVariantRegistrar registrar(task_id, name);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::register_task_variant<func>(registrar);
  }
  // cpu task      
  {
    if (cpu_func) {
      TaskVariantRegistrar registrar(task_id, name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::register_task_variant<cpu_func>(registrar);
    }
  }          
}

template <typename F>
void register_task(task_id_t task_id,
                   std::string const &name,
                   TaskSignature const & sig,
                   F const &func) {
  TaskSignature::task_sig_map.insert(task_id, sig);
  register_task<F>(task_id, name, func);
}

template <typename F>
void register_task(task_id_t task_id,
                   std::string const &name,
                   TaskSignature const & sig,
                   F const &func,
                   F const &cpu_func) {
  TaskSignature::task_sig_map.insert(task_id, sig);
  register_task<F>(task_id, name, func, cpu_func);
}

} // namespace FlexFlow
