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

#include "flexflow/model.h"

namespace FlexFlow {

using namespace Legion;

Loss::Loss(std::string const &loss, bool _repl_labels) {
  repl_labels = _repl_labels;
  if (loss == "categorical_crossentropy") {
    loss_type = LOSS_CATEGORICAL_CROSSENTROPY;
  } else if (loss == "sparse_categorical_crossentropy") {
    loss_type = LOSS_SPARSE_CATEGORICAL_CROSSENTROPY;
  } else if (loss == "mean_squared_error") {
    loss_type = LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE;
  } else if (loss == "identity") {
    loss_type = LOSS_IDENTITY;
  } else {
    // Unrecognized loss type
    assert(false);
  }
}

Loss::Loss(LossType _loss_type, bool _repl_labels)
    : loss_type(_loss_type), repl_labels(_repl_labels) {}

void Loss::backward(FFModel *model,
                    const ParallelTensor logit,
                    const ParallelTensor label) {
  int last_non_replica_dim = logit->num_dims - 1;
  while (logit->dims[last_non_replica_dim].is_replica_dim) {
    last_non_replica_dim -= 1;
  }
  // Compute scale factor for loss backpropagation
  if (loss_type == LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE) {
    assert(logit->get_volume() == label->get_volume());
    scale_factor = 2.0f / logit->get_volume();
  } else if (loss_type == LOSS_SPARSE_CATEGORICAL_CROSSENTROPY) {
    scale_factor = 1.0f;
  } else {
    scale_factor = 1.0f / model->config.batchSize;
  }
  // scale_factor = 1.0f;
  //  Use the same parallel strategy as the owner of logit
  std::string pcname = logit->owner_op->name;
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  Domain part_domain = runtime->get_index_space_domain(ctx, logit->parallel_is);
  Domain logit_domain = runtime->get_index_partition_color_space(
      ctx, logit->part.get_index_partition());
  Domain label_domain = runtime->get_index_partition_color_space(
      ctx, label->part.get_index_partition());
  if ((logit_domain != part_domain) || (label_domain != part_domain)) {
    fprintf(stderr,
            "Encounter inconsistency in parallelizing loss computation");
    assert(false);
  }
  ArgumentMap argmap;
  IndexLauncher launcher(LOSS_BWD_TASK_ID,
                         logit->parallel_is,
                         TaskArgument(this, sizeof(Loss)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         logit->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(logit->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    logit->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(
      logit->part, 0 /*projection id*/, READ_ONLY, EXCLUSIVE, logit->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(
      label->part, 0 /*projection id*/, READ_ONLY, EXCLUSIVE, label->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Loss::backward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    return backward_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template <int NDIM>
void Loss::backward_task_with_dim(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  Loss const *loss = (Loss *)task->args;

  if (loss->loss_type == LOSS_SPARSE_CATEGORICAL_CROSSENTROPY) {
    // sparse_categorical_crossentropy has label of dim: (batch_size, 1)
    TensorAccessorW<float, NDIM> acc_logit_grad(regions[0],
                                                task->regions[0],
                                                FID_DATA,
                                                ctx,
                                                runtime,
                                                true /*readOutput*/);
    TensorAccessorR<float, NDIM> acc_logit(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    TensorAccessorR<int, NDIM> acc_label(
        regions[2], task->regions[2], FID_DATA, ctx, runtime);
    // assertion the outter-most dim is replica dim and replica degree is 1
    assert(acc_logit.rect.hi[NDIM - 1] == acc_logit.rect.lo[NDIM - 1]);

    int num_classes = acc_logit.rect.hi[0] - acc_logit.rect.lo[0] + 1;
    int num_samples = acc_logit.rect.volume() / num_classes;
    // int num_samples =
    //     acc_logit.rect.hi[NDIM - 2] - acc_logit.rect.lo[NDIM - 2] + 1;
    // int num_classes = acc_logit.rect.volume() / num_samples;
    assert(acc_logit_grad.rect == acc_logit.rect);
    int k = 1;
    if (loss->repl_labels) {
      k = (acc_logit.rect.hi[NDIM - 1] - acc_logit.rect.lo[NDIM - 1] + 1) /
          (acc_label.rect.hi[NDIM - 1] - acc_label.rect.lo[NDIM - 1] + 1);
    }
    for (int i = 1; i < NDIM - 1; i++) {
      assert(acc_label.rect.hi[i] == acc_logit.rect.hi[i]);
      assert(acc_label.rect.lo[i] == acc_logit.rect.lo[i]);
    }
    assert(
        k * (acc_label.rect.hi[NDIM - 1] - acc_label.rect.lo[NDIM - 1] + 1) ==
        acc_logit.rect.hi[NDIM - 1] - acc_logit.rect.lo[NDIM - 1] + 1);
    assert(acc_label.rect.lo[0] == acc_label.rect.hi[0]);
    Loss::sparse_categorical_crossentropy_loss_backward_kernel_wrapper(
        acc_logit_grad.ptr,
        acc_logit.ptr,
        acc_label.ptr,
        acc_logit.rect.volume(),
        acc_logit_grad.rect.volume(),
        num_samples,
        num_classes,
        k,
        loss->scale_factor);
  } else {
    if (loss->repl_labels) {
      assert(false && "Loss not yet supported for aggr_spec.");
    }
    TensorAccessorW<float, NDIM> acc_logit_grad(regions[0],
                                                task->regions[0],
                                                FID_DATA,
                                                ctx,
                                                runtime,
                                                true /*readOutput*/);
    TensorAccessorR<float, NDIM> acc_logit(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    TensorAccessorR<float, NDIM> acc_label(
        regions[2], task->regions[2], FID_DATA, ctx, runtime);
    // other loss require label and logit have identical shape
    assert(acc_logit.rect == acc_label.rect);
    assert(acc_logit_grad.rect == acc_logit.rect);
    // assertion the outter-most dim is replica dim and replica degree is 1
    assert(acc_logit.rect.hi[NDIM - 1] == acc_logit.rect.lo[NDIM - 1]);
    int num_samples =
        acc_label.rect.hi[NDIM - 2] - acc_label.rect.lo[NDIM - 2] + 1;
    int num_channels = acc_logit.rect.volume() / num_samples;
    if (loss->loss_type == LOSS_CATEGORICAL_CROSSENTROPY) {
      Loss::categorical_crossentropy_loss_backward_kernel_wrapper(
          acc_logit_grad.ptr,
          acc_logit.ptr,
          acc_label.ptr,
          acc_logit.rect.volume(),
          acc_logit_grad.rect.volume(),
          loss->scale_factor);
    } else if (loss->loss_type == LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE) {
      Loss::mean_squared_error_avg_loss_backward_kernel_wrapper(
          acc_logit_grad.ptr,
          acc_logit.ptr,
          acc_label.ptr,
          acc_logit.rect.volume(),
          acc_logit_grad.rect.volume(),
          loss->scale_factor);
    } else if (loss->loss_type == LOSS_IDENTITY) {
      Loss::identity_loss_backward_kernel_wrapper(acc_logit_grad.ptr,
                                                  acc_logit.ptr,
                                                  acc_logit.rect.volume(),
                                                  acc_logit_grad.rect.volume(),
                                                  loss->scale_factor);
    } else {
      fprintf(stderr,
              "Unsupported loss --- report this error to the FlexFlow "
              "developers\n");
      assert(false);
    }
  }
}

}; // namespace FlexFlow
