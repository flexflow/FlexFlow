#include "kernels/initializer_kernels.h"
#include "kernels/accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/device.h"

namespace FlexFlow {

template <DataType DT>
struct ZeroInitKernel {
  void operator()(GenericTensorAccessorW const &tensor) const {
    auto arr = get<DT>(tensor);
    for (size_t i = 0; i < get_volume(tensor.shape); i++) {
      arr[i] = 0.0f;
    }
  }
};

void zero_init_kernel_cpu(GenericTensorAccessorW const &tensor) {
  DataTypeDispatch1<ZeroInitKernel>{}(tensor.data_type, tensor);
}

template <DataType DT>
struct ConstantInitKernel {
  void operator()(GenericTensorAccessorW const &tensor,
                  DataTypeValue value) const {
    auto arr = get<DT>(tensor);
    auto unwrapped_value = value.get<real_type_t<DT>>();
    for (size_t i = 0; i < get_volume(tensor.shape); i++) {
      arr[i] = unwrapped_value;
    }
  }
};

void constant_init_kernel_cpu(GenericTensorAccessorW const &tensor,
                              DataTypeValue value) {
  DataTypeDispatch1<ConstantInitKernel>{}(tensor.data_type, tensor, value);
}

void zero_init_kernel(TaskLocation const &loc,
                      GenericTensorAccessorW const &tensor) {
  if (loc == TaskLocation::CPU) {
    return zero_init_kernel_cpu(tensor);
  } else if (loc == TaskLocation::GPU) {
    return zero_init_kernel_gpu(tensor);
  }
}

void zero_init_kernel_gpu(GenericTensorAccessorW const &tensor) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
