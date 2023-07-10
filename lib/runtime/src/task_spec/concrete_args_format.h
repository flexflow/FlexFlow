#ifndef _FLEXFLOW_RUNTIME_SRC_CONCRETE_ARGS_FORMAT_H
#define _FLEXFLOW_RUNTIME_SRC_CONCRETE_ARGS_FORMAT_H

#include "legion.h"
#include "task_argument_accessor.h"
#include "runtime/task_spec/tensorless_task_invocation.h"
#include "utils/stack_map.h"

namespace FlexFlow {

/**
 * \class ConcreteArgsFormat
 * \brief struct that points to reserved bytes and processes concrete arguments
 * 
 * Deleted default constructor—must have all properties to create object; Used for processing;
 * Used in IndexArgsFormat, TaskArgumentsFormat, and Legion::TaskArgument;
 * Compiles to TaskArgumentsFormat and Legion::TaskArgument;
*/
struct ConcreteArgsFormat {
public:
  ConcreteArgsFormat() = delete;
  ConcreteArgsFormat(
      Legion::Serializer const &sez,
      TaskArgumentsFormat *reserved_bytes_for_fmt,
      stack_map<slot_id, TaskArgumentFormat, MAX_NUM_TASK_ARGUMENTS> const
          &fmts)
      : sez(sez), fmts(fmts) {}

public:
  Legion::Serializer sez;
  TaskArgumentsFormat *reserved_bytes_for_fmt;
  stack_map<slot_id, TaskArgumentFormat, MAX_NUM_TASK_ARGUMENTS> fmts;
};

/**
 * \fn ConcreteArgsFormat process_concrete_args(std::unordered_map<slot_id, ConcreteArgSpec> const &)
 * \param &specs An unordered_map of slot_id to ConcreteArgSpec
 * 
 * //Todo complete description 
*/
ConcreteArgsFormat process_concrete_args(std::unordered_map<slot_id, ConcreteArgSpec> const &specs);

/**
 * \fn ConcreteArgsFormat process_concrete_args(TensorlessTaskBinding const &binding)
 * \param &binding A TensorlessTaskBinding
 * \brief  Preceding function to process_concrete_args (std::unordered_map<...> const &specs)
 * 
 * Calls process_concrete_args given TensorlessTaskBinding
*/
ConcreteArgsFormat process_concrete_args(TensorlessTaskBinding const &binding);

}

#endif
