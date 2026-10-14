/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "internal/device.h"
#include "kernels/accessor.h"
#include "kernels/legion_ordered/legion_ordered_transform.h"
#include "kernels/transpose_kernels_gpu.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

struct TransposeStrides {
  int num_dim;
  int in_strides[MAX_TENSOR_DIM], out_strides[MAX_TENSOR_DIM],
      perm[MAX_TENSOR_DIM];
};

namespace Kernels {
namespace Transpose {

__global__ void transpose_simple_kernel(std::size_t volume,
                                        float const *in_ptr,
                                        float *out_ptr,
                                        const TransposeStrides info,
                                        float const beta) {
  CUDA_KERNEL_LOOP(o_idx, volume) {
    size_t i_idx = 0;
    size_t t = o_idx;
    for (int i = info.num_dim - 1; i >= 0; i--) {
      coord_t ratio = t / info.out_strides[i];
      t -= ratio * info.out_strides[i];
      i_idx += ratio * info.in_strides[info.perm[i]];
    }
    out_ptr[o_idx] += out_ptr[o_idx] * beta + in_ptr[i_idx];
  }
}

static TransposeStrides make_strides(TransposeAttrs const &m,
                                     TensorDims const &input_dims,
                                     TensorDims const &output_dims) {
  ASSERT(get_num_dims(input_dims) == m.permutation.num_tensor_dims());

  TransposeStrides info;
  num_tensor_dims_t num_dims = m.permutation.num_tensor_dims();
  info.num_dim = num_dims.int_from_num_tensor_dims();

  for (int i = 0; i < info.num_dim; i++) {
    legion_dim_t legion_dim = legion_dim_t{nonnegative_int{i}};
    ff_dim_t ff_dim = ff_dim_from_legion_dim(legion_dim, num_dims);

    if (i == 0) {
      info.in_strides[i] = 1;
      info.out_strides[i] = 1;
    } else {
      int in_dim_size =
          dim_at_idx(input_dims, legion_dim).int_from_positive_int();
      int out_dim_size =
          dim_at_idx(output_dims, legion_dim).int_from_positive_int();
      info.in_strides[i] = info.in_strides[i - 1] * in_dim_size;
      info.out_strides[i] = info.out_strides[i - 1] * out_dim_size;
    }

    ff_dim_t ff_permuted_dim = m.permutation.at_l(ff_dim);
    legion_dim_t legion_permuted_dim =
        legion_dim_from_ff_dim(ff_permuted_dim, num_dims);
    info.perm[i] = legion_permuted_dim.value.unwrap_nonnegative();
  }

  return info;
}

void gpu_forward_kernel(cudaStream_t stream,
                        TransposeAttrs const &m,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  transpose_simple_kernel<<<
      GET_BLOCKS(get_num_elements(output.shape.dims).int_from_positive_int()),
      CUDA_NUM_THREADS,
      0,
      stream>>>(get_num_elements(output.shape.dims).int_from_positive_int(),
                input.get_float_ptr(),
                output.get_float_ptr(),
                make_strides(m, input.shape.dims, output.shape.dims),
                /*beta=*/0.0f);
}

void gpu_backward_kernel(cudaStream_t stream,
                         TransposeAttrs const &m,
                         GenericTensorAccessorR const &out_grad,
                         GenericTensorAccessorW const &in_grad) {
  ASSERT(get_num_dims(in_grad.shape.dims) == m.permutation.num_tensor_dims());

  transpose_simple_kernel<<<
      GET_BLOCKS(get_num_elements(in_grad.shape.dims).int_from_positive_int()),
      CUDA_NUM_THREADS,
      0,
      stream>>>(get_num_elements(in_grad.shape.dims).int_from_positive_int(),
                out_grad.get_float_ptr(),
                in_grad.get_float_ptr(),
                make_strides(m, out_grad.shape.dims, in_grad.shape.dims),
                /*beta=*/1.0f);
}

} // namespace Transpose
} // namespace Kernels
} // namespace FlexFlow
