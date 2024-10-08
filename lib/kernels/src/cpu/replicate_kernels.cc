#include "kernels/datatype_dispatch.h"
#include "kernels/replicate_kernels_cpu.h"

namespace FlexFlow::Kernels::Replicate {

template <typename T>
void cpu_replicate_backward_kernel(T *input,
                                   T const *output,
                                   size_t num_elements,
                                   size_t num_replicas) {
  for (size_t i = 0; i < num_elements; i++) {
    T sum = 0;
    for (size_t j = 0; j < num_replicas; j++) {
      sum += output[i + j * num_elements];
    }
    input[i] = sum;
  }
}

template <DataType T>
struct CPUForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    memcpy(output.get<T>(),
           input.get<T>(),
           input.shape.num_elements() * size_of_datatype(T));
  }
};

template <DataType T>
struct CPUBackwardKernel {
  void operator()(GenericTensorAccessorW const &input,
                  GenericTensorAccessorR const &output,
                  size_t num_replicas) {
    cpu_replicate_backward_kernel(input.get<T>(),
                                  output.get<T>(),
                                  input.shape.num_elements(),
                                  num_replicas);
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  DataTypeDispatch1<CPUForwardKernel>{}(input.data_type, input, output);
}

void cpu_backward_kernel(GenericTensorAccessorW const &input,
                         GenericTensorAccessorR const &output,
                         size_t num_replicas) {
  DataTypeDispatch1<CPUBackwardKernel>{}(
      input.data_type, input, output, num_replicas);
}

} // namespace FlexFlow::Kernels::Replicate
