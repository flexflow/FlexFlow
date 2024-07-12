#include "kernels/cast_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"

namespace FlexFlow {
namespace Kernels {
namespace Cast {
namespace CPU {

template <typename IDT, typename ODT>
void cast_forward(IDT const *input, ODT *output, size_t volume) {
  for (size_t i = 0; i < volume; ++i) {
    output[i] = static_cast<ODT>(input[i]);
  }
}

template <typename IDT, typename ODT>
void cast_backward(IDT const *input, ODT *output, size_t volume, ODT beta) {
  for (size_t i = 0; i < volume; i++) {
    output[i] = static_cast<ODT>(input[i]) + beta * output[i];
  }
}

template <DataType IDT, DataType ODT>
struct ForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    size_t volume = input.shape.get_volume();
    cast_forward(input.get<IDT>(), output.get<ODT>(), volume);
  }
};

template <DataType IDT, DataType ODT>
struct BackwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    size_t volume = input.shape.get_volume();
    cast_backward(
        input.get<IDT>(), output.get<ODT>(), volume, cast_to<ODT>(1.0f));
  }
};

void forward_kernel(GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    DataType input_type,
                    DataType output_type) {
  DataTypeDispatch2<ForwardKernel>{}(input_type, output_type, input, output);
}

void backward_kernel(GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &output,
                     DataType input_type,
                     DataType output_type) {
  DataTypeDispatch2<BackwardKernel>{}(input_type, output_type, input, output);
}

} // namespace CPU
} // namespace Cast
} // namespace Kernels
} // namespace FlexFlow
