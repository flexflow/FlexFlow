#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_GROUPBY_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_GROUPBY_H


namespace FlexFlow {

class GroupByPerDeviceState : public PerDeviceOpState {
public:
  GroupByPerDeviceState(FFHandler handle, int n);
  ~GroupByPerDeviceState(void);
  float **dev_region_ptrs;
};

} // namespace FlexFlow

#endif
