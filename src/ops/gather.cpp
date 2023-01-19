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

#include "flexflow/ops/gather.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

GatherMeta::GatherMeta(FFHandler handler, Gather const *gather)
    : OpMeta(handler, gather) {
  legion_dim = gather->legion_dim;
}

void Gather::forward_kernel_wrapper(GatherMeta const *m,
                                    GenericTensorAccessorR const &input,
                                    GenericTensorAccessorR const &index,
                                    GenericTensorAccessorW const &output) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  coord_t stride = 1;
  for (int i = 0; i < m->legion_dim; i++) {
    stride *= (output.domain.hi()[i] - output.domain.lo()[i] + 1);
  }
  coord_t dim_size =
      output.domain.hi()[m->legion_dim] - output.domain.lo()[m->legion_dim] + 1;
  if (index.data_type == DT_INT32) {
    Gather::forward_kernel(input.get_float_ptr(),
                           index.get_int32_ptr(),
                           output.get_float_ptr(),
                           output.domain.get_volume(),
                           stride,
                           dim_size,
                           stream);
  } else {
    assert(index.data_type == DT_INT64);
    Gather::forward_kernel(input.get_float_ptr(),
                           index.get_int64_ptr(),
                           output.get_float_ptr(),
                           output.domain.get_volume(),
                           stride,
                           dim_size,
                           stream);
  }
}

template <typename TI>
void Gather::forward_kernel(float const *input_ptr,
                            TI const *index_ptr,
                            float *output_ptr,
                            coord_t output_size,
                            coord_t stride,
                            coord_t dim_size,
                            hipStream_t stream) {
  assert(input_ptr != nullptr);
  assert(index_ptr != nullptr);
  assert(output_ptr != nullptr);
  assert(false && "hip kernel to be implemented");
}

void Gather::backward_kernel_wrapper(GatherMeta const *m,
                                     GenericTensorAccessorR const &output_grad,
                                     GenericTensorAccessorR const &index,
                                     GenericTensorAccessorW const &input_grad) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  coord_t stride = 1;
  for (int i = 0; i < m->legion_dim; i++) {
    stride *= (output_grad.domain.hi()[i] - output_grad.domain.lo()[i] + 1);
  }
  coord_t dim_size = output_grad.domain.hi()[m->legion_dim] -
                     output_grad.domain.lo()[m->legion_dim] + 1;
  if (index.data_type == DT_INT32) {
    Gather::backward_kernel(output_grad.get_float_ptr(),
                            index.get_int32_ptr(),
                            input_grad.get_float_ptr(),
                            output_grad.domain.get_volume(),
                            stride,
                            dim_size,
                            stream);
  } else {
    assert(index.data_type == DT_INT64);
    Gather::backward_kernel(output_grad.get_float_ptr(),
                            index.get_int64_ptr(),
                            input_grad.get_float_ptr(),
                            output_grad.domain.get_volume(),
                            stride,
                            dim_size,
                            stream);
  }
}

template <typename TI>
void Gather::backward_kernel(float const *output_grad_ptr,
                             TI const *index_ptr,
                             float *input_grad_ptr,
                             coord_t output_size,
                             coord_t stride,
                             coord_t dim_size,
                             hipStream_t stream) {
  assert(output_grad_ptr != nullptr);
  assert(input_grad_ptr != nullptr);
  assert(index_ptr != nullptr);
  assert(false && "hip kernel to be implemented");
}

}; // namespace FlexFlow
