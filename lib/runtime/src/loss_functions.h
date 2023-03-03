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

#ifndef _FF_LOSS_FUNCTIONS_H_
#define _FF_LOSS_FUNCTIONS_H_

#include "ffconst.h"
#include "legion.h"
#include "parallel_tensor.h"

namespace FlexFlow {

class FFModel;

class Loss {
public:
  Loss(std::string const &loss, bool _repl_labels = false);
  Loss(LossType _loss_type, bool _repl_labels = false);

  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <int NDIM>
  static void
      backward_task_with_dim(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  void backward(FFModel *model,
                const ParallelTensor logit,
                const ParallelTensor label);
  template <int NDIM>
  void backward_with_dim(FFModel *model,
                         const ParallelTensor logit,
                         const ParallelTensor label);
  static void sparse_categorical_crossentropy_loss_backward_kernel_wrapper(
      float *logit_grad_ptr,
      float const *logit_ptr,
      int const *label_ptr,
      size_t logit_volume,
      size_t logit_grad_volume,
      int num_samples,
      int num_classes,
      int k,
      float scale_factor);
  static void categorical_crossentropy_loss_backward_kernel_wrapper(
      float *logit_grad_ptr,
      float const *logit_ptr,
      float const *label_ptr,
      size_t logit_volume,
      size_t logit_grad_volume,
      float scale_factor);
  static void mean_squared_error_avg_loss_backward_kernel_wrapper(
      float *logit_grad_ptr,
      float const *logit_ptr,
      float const *label_ptr,
      size_t logit_volume,
      size_t logit_grad_volume,
      float scale_factor);
  static void identity_loss_backward_kernel_wrapper(float *loss_grad_ptr,
                                                    float const *loss_ptr,
                                                    size_t loss_volume,
                                                    size_t loss_grad_volume,
                                                    float scale_factor);

public:
  FFModel *model;
  LossType loss_type;
  bool repl_labels; // for aggregate_spec: More predictions than labels
  // scale factor for computing the logit gradients
  // normally 1.0f / global_batch_size
  float scale_factor;
};

}; // namespace FlexFlow

#endif
