#include "kernels/combine_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"

namespace FlexFlow {
namespace Kernels {
namespace Combine {
namespace CPU {

template <DataType DT>
struct ForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    memcpy(output.get<DT>(),
           input.get<DT>(),
           input.shape.get_volume() * size_of_datatype(DT));
  }
};

template <DataType DT>
struct BackwardKernel {
  void operator()(GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorW const &input_grad) {
    size_t num_elements = output_grad.shape.get_volume();
    for (int i = 0; i < num_elements; ++i) {
      input_grad.get<DT>()[i] += output_grad.get<DT>()[i];
    }
  }
};

void forward_kernel(GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  DataTypeDispatch1<ForwardKernel>{}(input.data_type, input, output);
}

void backward_kernel(GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad) {
  DataTypeDispatch1<BackwardKernel>{}(
      input_grad.data_type, output_grad, input_grad);
}

} // namespace CPU
} // namespace Combine
} // namespace Kernels
} // namespace FlexFlow
