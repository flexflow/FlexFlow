#include "kernels/cast_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"

namespace FlexFlow::Kernels::Cast {

template <typename IDT, typename ODT>
void cpu_cast_forward(IDT const *input, ODT *output, size_t volume) {
  for (size_t i = 0; i < volume; ++i) {
    output[i] = static_cast<ODT>(input[i]);
  }
}

template <typename IDT, typename ODT>
void cpu_cast_backward(IDT const *input, ODT *output, size_t volume, ODT beta) {
  for (size_t i = 0; i < volume; i++) {
    output[i] = static_cast<ODT>(input[i]) + beta * output[i];
  }
}

template <DataType IDT, DataType ODT>
struct CPUForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    size_t volume = input.shape.get_volume();
    cpu_cast_forward(input.get<IDT>(), output.get<ODT>(), volume);
  }
};

template <DataType IDT, DataType ODT>
struct CPUBackwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    size_t volume = input.shape.get_volume();
    cpu_cast_backward(
        input.get<IDT>(), output.get<ODT>(), volume, cast_to<ODT>(1.0f));
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        DataType input_type,
                        DataType output_type) {
  DataTypeDispatch2<CPUForwardKernel>{}(input_type, output_type, input, output);
}

void cpu_backward_kernel(GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &output,
                         DataType input_type,
                         DataType output_type) {
  DataTypeDispatch2<CPUBackwardKernel>{}(
      input_type, output_type, input, output);
}

} // namespace FlexFlow::Kernels::Cast
