#ifndef _FLEXFLOW_KERNELS_PER_NODE_OP_STATE_H
#define _FLEXFLOW_KERNELS_PER_NODE_OP_STATE_H

#include "kernels/ff_handler.h"
#include "op-attrs/ffconst.h"
#include <vector>
#include "utils/stack_vector.h"
#include <type_traits>

namespace FlexFlow {

class Op;

class PerDeviceOpState {
public:
  PerDeviceOpState(FFHandler handle);
  PerDeviceOpState(FFHandler handle, bool profiling);

  template <typename C>
  PerDeviceOpState(FFHandler handle, 
                   C const &op_input_types, 
                   C const &op_weight_types,
                   C const &op_output_types) 
    : handle(handle),
      input_type(op_input_types.begin(), op_input_types.end()),
      weight_type(op_weight_types.begin(), op_weight_types.end()),
      output_type(op_output_types.begin(), op_output_types.end())
    { 
      static_assert(std::is_same<typename C::value_type, DataType>::value, "Invalid data type in container");
    }

public:
  FFHandler handle;
  bool profiling; // Measure the run time of the task

  stack_vector<bool, MAX_NUM_INPUTS> trainableInputs;
  stack_vector<DataType, MAX_NUM_INPUTS> input_type;
  stack_vector<DataType, MAX_NUM_WEIGHTS> weight_type;
  stack_vector<DataType, MAX_NUM_OUTPUTS> output_type;
};

}; 

#endif 
