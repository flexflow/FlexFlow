#ifndef _OP_META_H
#define _OP_META_H

#include "flexflow/config.h"

namespace FlexFlow {

class Op;

class OpMeta {
public:
  OpMeta(FFHandler _handle);
  OpMeta(FFHandler _handle, Op const *op);

public:
  FFHandler handle;
  bool profiling; // Measure the run time of the task
  bool inference_debugging;
  int decoding_step;
  char op_name[MAX_OPNAME];
  LayerID layer_guid;
  bool trainableInputs[MAX_NUM_INPUTS];
  DataType input_type[MAX_NUM_INPUTS];
  DataType weight_type[MAX_NUM_WEIGHTS];
  DataType output_type[MAX_NUM_OUTPUTS];
};

}; // namespace FlexFlow

#endif //_OP_META_H
