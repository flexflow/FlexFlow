#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_GROUPBY_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_GROUPBY_H

#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class GroupByMeta : public PerDeviceOpState {
public:
  GroupByMeta(FFHandler handle, int n);
  ~GroupByMeta(void);
  float **dev_region_ptrs;
};


}

#endif 
