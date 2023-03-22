#include "utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

using namespace Legion;

template <typename DT>
__global__ void zero_array(DT *ptr, coord_t size) {
  CUDA_KERNEL_LOOP(i, size) {
    ptr[i] = 0;
  }
}

}; // namespace FlexFlow
