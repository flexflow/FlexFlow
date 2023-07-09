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

#include "loss_functions.h"
#include "kernels/loss_function_kernels.h"
#include "legion.h"
#include "runtime/profiling.h"
#include "task_spec/task_argument_accessor.h"

namespace FlexFlow {

enum LossSlots {
  LOGIT_GRAD,
  LOGIT,
  LABEL,
  LOSS_ATTRS,
  BATCH_SIZE,
  PROFILING_SETTINGS
};

TaskInvocation backward_invocation(LossAttrs const &attrs,
                                   EnableProfiling enable_profiling,
                                   parallel_tensor_guid_t logit,
                                   parallel_tensor_guid_t label) {
  auto binding = IndexTaskBinding{LOGIT};
  StandardTypedTaskArg<LossAttrs> arg = attrs;
  binding.bind_arg(LOSS_ATTRS, attrs);
  binding.bind(LOGIT, logit);
  binding.bind(LABEL, label);
  binding.bind(LOGIT_GRAD, grad(logit));
  binding.bind_arg(PROFILING_SETTINGS, profiling_settings());

  /* if ((logit_domain != part_domain) || (label_domain != part_domain)) { */ // TODO @lockshaw make sure this is still checked
  /*   fprintf(stderr, */
  /*           "Encounter inconsistency in parallelizing loss computation"); */
  /*   assert(false); */
  /* } */
  return {LOSS_BWD_TASK_ID, binding};
}

static void
    loss_backward_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  auto attrs = acc.get_argument<LossAttrs>(LOSS_ATTRS);
  auto profiling_settings =
      acc.get_argument<ProfilingSettings>(PROFILING_SETTINGS);
  auto batch_size = acc.get_argument<int>(BATCH_SIZE);
  auto logit_grad = acc.get_tensor<Permissions::RW>(LOGIT_GRAD);
  auto logit = acc.get_tensor<Permissions::RO>(LOGIT);
  auto label = acc.get_tensor<Permissions::RO>(LABEL);

  LossFunction loss_type = get_loss_function(attrs);
  float scale_factor = 1.0f / batch_size;
  if (loss_type == LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE) {
    assert(logit.shape.get_volume() == label.shape.get_volume());
    scale_factor = 2.0f / logit.shape.get_volume();
  }

  if (loss_type == LossFunction::SPARSE_CATEGORICAL_CROSSENTROPY) {
    // assertion the outter-most dim is replica dim and replica degree is 1
    auto scce_attrs = get<SparseCategoricalCrossEntropyLossAttrs>(attrs);
    size_t ndim = logit.shape.num_dims();
    assert(logit.shape.at(legion_dim_t(ndim - 1)) == 1);
    int num_samples = logit.shape.at(legion_dim_t(ndim - 2));
    int num_classes = logit.shape.get_volume() / num_samples;
    assert(logit_grad.shape == logit.shape);
    int k = 1;
    if (scce_attrs.replace_labels) {
      k = logit.shape.at(legion_dim_t(ndim - 1)) /
          label.shape.at(legion_dim_t(
              ndim - 1)); // TODO FIXME something seems wrong here, isn't the
                          // numerator guaranteed to be 1?
    }
    assert(label.shape.sub_shape(legion_dim_t(1), nullopt) ==
           logit.shape.sub_shape(legion_dim_t(1), nullopt));
    assert(k * label.shape.at(legion_dim_t(ndim - 1)) ==
           logit.shape.at(legion_dim_t(ndim - 1)));
    assert(label.shape.at(legion_dim_t(0)) == 1);

    profile(sparse_categorical_crossentropy_loss_backward_kernel,
            profiling_settings,
            "[SparseCategoricalCrossEntropyLoss] backward_time = %.2lfms\n",
            get_float_ptr(logit_grad),
            get_float_ptr(logit),
            get_int32_ptr(label),
            logit.shape.get_volume(),
            get_volume(logit_grad.shape),
            num_samples,
            num_classes,
            k,
            scale_factor);
  } else {
    assert(logit.shape == label.shape);
    assert(logit_grad.shape == logit.shape);
    // assertion the outter-most dim is replica dim and replica degree is 1
    size_t ndim = logit.shape.num_dims();
    assert(logit.shape.at(legion_dim_t(ndim - 1)) == 1);
    int num_samples = label.shape.at(legion_dim_t(ndim - 1));
    int num_channels = logit.shape.get_volume() / num_samples;
    switch (loss_type) {
      case LossFunction::CATEGORICAL_CROSSENTROPY: {
        profile(categorical_crossentropy_loss_backward_kernel,
                profiling_settings,
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
                profiling_settings,
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
                profiling_settings,
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

template <>
void register_task<LOSS_BWD_TASK_ID>() {
  TaskSignature sig;
  sig.add_arg_slot<LossAttrs>(LOSS_ATTRS);
  sig.add_arg_slot<ProfilingSettings>(PROFILING_SETTINGS);
  sig.add_slot(LOGIT, {SlotType::TENSOR, Permissions::RO});
  sig.add_slot(LABEL, {SlotType::TENSOR, Permissions::RO});
  sig.add_slot(LOGIT_GRAD, {SlotType::TENSOR, Permissions::RW});

  register_task(LOSS_BWD_TASK_ID, "Loss Backward", sig, loss_backward_task);
}

} // namespace FlexFlow
