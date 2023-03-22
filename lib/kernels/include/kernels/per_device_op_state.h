#ifndef _FLEXFLOW_KERNELS_PER_NODE_OP_STATE_H
#define _FLEXFLOW_KERNELS_PER_NODE_OP_STATE_H

#include "kernels/config.h"
#include "op-attrs/ffconst.h"

namespace FlexFlow {

class Op;

class PerDeviceOpState {
public:
  PerDeviceOpState(FFHandler handle);
  PerDeviceOpState(FFHandler handle, bool profiling);
  PerDeviceOpState(FFHandler handle, Op const *op);

public:
  FFHandler handle;
  bool profiling; // Measure the run time of the task
  bool trainableInputs[MAX_NUM_INPUTS];
  DataType input_type[MAX_NUM_INPUTS];
  DataType weight_type[MAX_NUM_WEIGHTS];
  DataType output_type[MAX_NUM_OUTPUTS];
};

}; 

#endif 
