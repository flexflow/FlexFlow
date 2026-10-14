#include "kernels/optimizer_kernels_cpu.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/tensor_accessor_binary_ops.h"
#include "kernels/tensor_accessor_unary_ops.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void cpu_sgd_update_task(float lr,
                         float momentum,
                         bool nesterov,
                         float weight_decay,
                         GenericTensorAccessorR const &weight_grad,
                         GenericTensorAccessorW const &weight,
                         std::optional<GenericTensorAccessorW> const &sgd_v) {
  // based on sgd_update in lib/kernels/src/cuda/optimizer_kernels.cu

  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  std::cerr << "weight_grad=" << format_accessor_r_contents(weight_grad)
            << std::endl
            << "weight=" << format_accessor_w_contents(weight) << std::endl;

  GenericTensorAccessorW gt = tensor_accessor_elementwise_add(
      weight_grad,
      read_only_accessor_from_write_accessor(tensor_accessor_scale_by_constant(
          read_only_accessor_from_write_accessor(weight),
          weight_decay,
          cpu_allocator)),
      cpu_allocator);

  if (momentum > 0.0f) {
    tensor_accessor_scale_by_constant_inplace(sgd_v.value(), momentum);
    tensor_accessor_elementwise_add_to(
        read_only_accessor_from_write_accessor(sgd_v.value()),
        read_only_accessor_from_write_accessor(gt),
        sgd_v.value());

    if (nesterov) {
      tensor_accessor_elementwise_add_to(
          read_only_accessor_from_write_accessor(gt),
          read_only_accessor_from_write_accessor(
              tensor_accessor_scale_by_constant(
                  read_only_accessor_from_write_accessor(sgd_v.value()),
                  momentum,
                  cpu_allocator)),
          gt);
    } else {
      copy_accessor_data_to_l_from_r(
          gt, read_only_accessor_from_write_accessor(sgd_v.value()));
    }
  }

  tensor_accessor_elementwise_subtract_to(
      read_only_accessor_from_write_accessor(weight),
      read_only_accessor_from_write_accessor(tensor_accessor_scale_by_constant(
          read_only_accessor_from_write_accessor(gt), lr, cpu_allocator)),
      weight);
}

void cpu_adam_update_task(float alpha_t,
                          float beta1,
                          float beta2,
                          float weight_decay,
                          float epsilon,
                          float const *weight_grad_ptr,
                          size_t size,
                          int num_replicas,
                          float *weight_ptr,
                          float *adam_v_ptr,
                          float *adam_m_ptr) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
