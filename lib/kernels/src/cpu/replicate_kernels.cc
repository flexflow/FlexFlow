#include "kernels/datatype_dispatch.h"
#include "kernels/replicate_kernels_cpu.h"

namespace FlexFlow::Kernels::Replicate {

template <DataType DT>
struct CPUForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output) {
    memcpy(output.get<DT>(),
           input.get<DT>(),
           input.shape.num_elements() * size_of_datatype(DT));
  }
};

template <DataType DT>
struct CPUBackwardKernel {
  void operator()(GenericTensorAccessorR const &output,
                  GenericTensorAccessorW &input,
                  size_t num_replicas) {
    using T = real_type_t<DT>;
    for (size_t i = 0; i < input.shape.num_elements(); i++) {
      T cur_sum = 0;
      for (size_t j = 0; j < num_replicas; j++) {
        cur_sum += output.at<DT>(i, j);
      }
      input.at<DT>(i) = cur_sum;
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW &output) {
  DataTypeDispatch1<CPUForwardKernel>{}(
      input.data_type, input, std::ref(output));
}

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorW &input,
                         size_t num_replicas) {
  DataTypeDispatch1<CPUBackwardKernel>{}(
      input.data_type, output, std::ref(input), num_replicas);
}

} // namespace FlexFlow::Kernels::Replicate
