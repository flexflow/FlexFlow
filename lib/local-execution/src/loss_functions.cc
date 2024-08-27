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

#include "op-attrs/ops/loss_functions.h"
#include "kernels/loss_function_kernels.h"
#include "local-execution/loss_functions.h"
#include "local-execution/profiling.h"

namespace FlexFlow {

enum Slots { LOGIT, LABEL, ATTRS, PROFILING };

TaskSignature get_loss_bwd_signature() {
  TaskSignature sig = make_empty_task_signature();
  add_slot(sig, LOGIT, IsGrad::NO);
  add_slot(sig, LABEL, IsGrad::NO);
  add_slot(sig, LOGIT, IsGrad::YES);
  add_arg_slot<LossAttrs>(sig, ATTRS);
  add_arg_slot<ProfilingSettings>(sig, PROFILING);
  return sig;
}

TaskInvocation
    backward(LossAttrs const &attrs, tensor_guid_t logit, tensor_guid_t label) {
  TaskBinding b;
  b.bind(LOGIT, TensorGuidSpec{logit, IsGrad::NO});
  b.bind(LABEL, TensorGuidSpec{label, IsGrad::NO});
  b.bind(LOGIT, TensorGuidSpec{logit, IsGrad::YES});
  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());

  return {task_id_t::LOSS_BWD_TASK_ID, b};
}

static void backward_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<LossAttrs>(ATTRS);
  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto logit_grad = acc.get_tensor_grad<Permissions::RW>(LOGIT);
  auto logit = acc.get_tensor<Permissions::RO>(LOGIT);
  auto label = acc.get_tensor<Permissions::RO>(LABEL);
  int batch_size = logit.shape.at(legion_dim_t{1});
  // assuming logit shape is [parallel dim(?), batch dim, num classes]

  LossFunction loss_type = get_loss_function(attrs);
  float scale_factor = 1.0f / batch_size;
  if (loss_type == LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE) {
    assert(logit.shape.get_volume() == label.shape.get_volume());
    scale_factor = 2.0f / logit.shape.get_volume();
  }

  if (loss_type == LossFunction::SPARSE_CATEGORICAL_CROSSENTROPY) {
    // label shape is [parallel dim(?), batch dim, 1]
    auto scce_attrs = attrs.get<SparseCategoricalCrossEntropyLossAttrs>();
    size_t ndim = logit.shape.num_dims();
    int num_classes = logit.shape.at(legion_dim_t{0});
    assert(logit_grad.shape == logit.shape);
    int k = 1;
    if (scce_attrs.replace_labels) {
      k = logit.shape.at(legion_dim_t(ndim - 1)) /
          label.shape.at(legion_dim_t(
              ndim - 1)); // TODO FIXME something seems wrong here, isn't the
                          // numerator guaranteed to be 1? <--- this is not the
                          // case because of the potential parallel dim
    }
    assert(label.shape.sub_shape(legion_dim_t(1), std::nullopt) ==
           logit.shape.sub_shape(legion_dim_t(1), std::nullopt));
    assert(k * label.shape.at(legion_dim_t(ndim - 1)) ==
           logit.shape.at(legion_dim_t(ndim - 1)));
    assert(label.shape.at(legion_dim_t(0)) == 1);

    profile(sparse_categorical_crossentropy_loss_backward_kernel,
            profiling,
            "[SparseCategoricalCrossEntropyLoss] backward_time = %.2lfms\n",
            get_float_ptr(logit_grad),
            get_float_ptr(logit),
            reinterpret_cast<int const *>(get_float_ptr(label)),
            get_volume(logit.shape),
            get_volume(logit_grad.shape),
            batch_size,
            num_classes,
            k,
            scale_factor);
  } else {
    assert(logit.shape == label.shape);
    assert(logit_grad.shape == logit.shape);
    int num_channels = logit.shape.at(legion_dim_t{0});
    switch (loss_type) {
      case LossFunction::CATEGORICAL_CROSSENTROPY: {
        profile(categorical_crossentropy_loss_backward_kernel,
                profiling,
                "[CategoricalCrossEntropyLoss] backward_time = %.2lfms\n",
                get_float_ptr(logit_grad),
                get_float_ptr(logit),
                get_float_ptr(label),
                get_volume(logit.shape),
                get_volume(logit_grad.shape),
                scale_factor);
        break;
      }
      case LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE: {
        profile(mean_squared_error_avg_loss_backward_kernel,
                profiling,
                "[MeanSquaredErrorAvgLoss] backward_time = %.2lfms\n",
                get_float_ptr(logit_grad),
                get_float_ptr(logit),
                get_float_ptr(label),
                get_volume(logit.shape),
                get_volume(logit_grad.shape),
                scale_factor);
        break;
      }
      case LossFunction::IDENTITY: {
        profile(identity_loss_backward_kernel,
                profiling,
                "[IdentityLoss] backward_time = %.2lfms\n",
                get_float_ptr(logit_grad),
                get_float_ptr(logit),
                get_volume(logit.shape),
                get_volume(logit_grad.shape),
                scale_factor);
        break;
      }
      default:
        throw mk_runtime_error(
            "Unsupported loss function {}. Please report this as an issue.",
            loss_type);
    }
  }
}

TaskImplFunction get_loss_bwd_task_impl() {
  return TaskImplFunction{GenericTaskImplFunction{backward_task_impl}};
}

} // namespace FlexFlow
