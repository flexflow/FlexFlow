#ifndef _OP_META_H
#define _OP_META_H

#include "flexflow/config.h"

namespace FlexFlow {

class OpMeta {
public:
  OpMeta(FFHandler _handle);

public:
  FFHandler handle;
  bool profiling; // Measure the run time of the task
  bool trainableInputs[MAX_NUM_INPUTS];
};

}; // namespace FlexFlow

#endif //_OP_META_H