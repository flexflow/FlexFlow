#include "flexflow/op_meta.h"

namespace FlexFlow {

OpMeta::OpMeta(FFHandler _handle) : handle(_handle), profiling(false) {
  for (int i = 0; i < MAX_NUM_INPUTS; i++)
    trainableInputs[i] = true;
}

}  // namespace FlexFlow
