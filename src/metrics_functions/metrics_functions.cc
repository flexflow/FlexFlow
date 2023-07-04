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

#include "flexflow/metrics_functions.h"
#include "flexflow/model.h"
#include <iostream>

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

Metrics::Metrics(LossType _loss_type, std::vector<MetricsType> const &metrics)
    : loss_type(_loss_type), measure_accuracy(false),
      measure_categorical_crossentropy(false),
      measure_sparse_categorical_crossentropy(false),
      measure_mean_squared_error(false), measure_root_mean_squared_error(false),
      measure_mean_absolute_error(false) {
  for (size_t i = 0; i < metrics.size(); i++) {
    switch (metrics[i]) {
      case METRICS_ACCURACY:
        measure_accuracy = true;
        continue;
      case METRICS_CATEGORICAL_CROSSENTROPY:
        measure_categorical_crossentropy = true;
        continue;
      case METRICS_SPARSE_CATEGORICAL_CROSSENTROPY:
        measure_sparse_categorical_crossentropy = true;
        continue;
      case METRICS_MEAN_SQUARED_ERROR:
        measure_mean_squared_error = true;
        continue;
      case METRICS_ROOT_MEAN_SQUARED_ERROR:
        measure_root_mean_squared_error = true;
        continue;
      case METRICS_MEAN_ABSOLUTE_ERROR:
        measure_mean_absolute_error = true;
        continue;
      default:
        fprintf(stderr, "Unrecogonized metrics type\n");
        assert(false);
    }
  }
}

void Metrics::compute(FFModel *model,
                      const ParallelTensor logit,
                      const ParallelTensor label) {
  // Use the same parallel strategy as the owner of logit
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  Domain part_domain = runtime->get_index_space_domain(ctx, logit->parallel_is);
  Domain logit_domain = runtime->get_index_partition_color_space(
      ctx, logit->part.get_index_partition());
  Domain label_domain = runtime->get_index_partition_color_space(
      ctx, label->part.get_index_partition());
  if ((logit_domain != part_domain) || (label_domain != part_domain)) {
    fprintf(stderr,
            "Encounter inconsistency in parallelizing loss computation\n");
    assert(false);
  }
  ArgumentMap argmap;
  IndexLauncher launcher(METRICS_COMP_TASK_ID,
                         logit->parallel_is,
                         TaskArgument(this, sizeof(Metrics)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         logit->machine_view.hash());
  // std::cout << "logit shape: " << logit->get_shape() << std::endl;
  // std::cout << "label shape: " << label->get_shape() << std::endl;
  launcher.add_region_requirement(RegionRequirement(
      logit->part, 0 /*projection id*/, READ_ONLY, EXCLUSIVE, logit->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(
      label->part, 0 /*projection id*/, READ_ONLY, EXCLUSIVE, label->region));
  launcher.add_field(1, FID_DATA);
  FutureMap new_metrics = runtime->execute_index_space(ctx, launcher);
  // Update metrics
  TaskLauncher metrics_task(UPDATE_METRICS_TASK_ID,
                            TaskArgument(this, sizeof(Metrics)));
  metrics_task.add_future(model->current_metrics);
  for (Domain::DomainPointIterator it(part_domain); it; it++) {
    metrics_task.add_future(new_metrics[*it]);
  }
  model->current_metrics = runtime->execute_task(ctx, metrics_task);
}

PerfMetrics Metrics::compute_task(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    return compute_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  PerfMetrics invalid;
  return invalid;
}

template <int NDIM>
PerfMetrics
    Metrics::compute_task_with_dim(Task const *task,
                                   std::vector<PhysicalRegion> const &regions,
                                   Context ctx,
                                   Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Metrics const *me = (Metrics *)task->args;
  PerfMetrics perf_zc;

  if (me->loss_type == LOSS_SPARSE_CATEGORICAL_CROSSENTROPY) {
    TensorAccessorR<float, NDIM> acc_logit(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    TensorAccessorR<int, NDIM> acc_label(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    // assume that the leading dim is replica dim
    assert(acc_logit.rect.hi[NDIM - 1] == acc_logit.rect.lo[NDIM - 1]);
    int num_effective_samples = acc_label.rect.volume();
    int num_classes = acc_logit.rect.hi[0] - acc_logit.rect.lo[0] + 1;
    assert(num_effective_samples * num_classes == acc_logit.rect.volume());
    for (int i = 1; i < NDIM; i++) {
      assert(acc_label.rect.hi[i] == acc_logit.rect.hi[i]);
      assert(acc_label.rect.lo[i] == acc_logit.rect.lo[i]);
    }
    assert(acc_label.rect.lo[0] == acc_label.rect.hi[0]);
    // Cannot measure categorical_crossentropy w/ sparse labels
    // Use measure_sparse_categorical_crossentropy instead
    // std::cout << "num_classes: " << num_classes << std::endl;
    assert(!me->measure_categorical_crossentropy);
    Metrics::update_metrics_sparse_label_kernel_wrapper(acc_logit.ptr,
                                                        acc_label.ptr,
                                                        me,
                                                        num_effective_samples,
                                                        num_classes,
                                                        perf_zc);
  } else {
    TensorAccessorR<float, NDIM> acc_logit(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    TensorAccessorR<float, NDIM> acc_label(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    // other loss require label and logit have identical shape
    assert(acc_logit.rect == acc_label.rect);
    // assume that the leading dim is replica dim
    assert(acc_logit.rect.hi[NDIM - 1] == acc_logit.rect.lo[NDIM - 1]);
    int num_samples =
        acc_logit.rect.hi[NDIM - 2] - acc_logit.rect.lo[NDIM - 2] + 1;
    int num_classes = acc_logit.rect.volume() / num_samples;
    // Use CUDA_NUM_THREADS may result in out of resources so we set
    // #threads=256
    Metrics::update_metrics_label_kernel_wrapper(
        acc_logit.ptr, acc_label.ptr, me, num_samples, num_classes, perf_zc);
  }
  return perf_zc;
}

PerfMetrics::PerfMetrics(void)
    : train_all(0), train_correct(0), cce_loss(0.0f), sparse_cce_loss(0.0f),
      mse_loss(0.0f), rmse_loss(0.0f), mae_loss(0.0f) {
  start_time = Realm::Clock::current_time_in_microseconds();
}

void PerfMetrics::update(PerfMetrics const &one) {
  train_all += one.train_all;
  train_correct += one.train_correct;
  cce_loss += one.cce_loss;
  sparse_cce_loss += one.sparse_cce_loss;
  mse_loss += one.mse_loss;
  rmse_loss += one.rmse_loss;
  mae_loss += one.mae_loss;
}

void PerfMetrics::apply_scale(float scale) {
  cce_loss *= scale;
  sparse_cce_loss *= scale;
  mse_loss *= scale;
  rmse_loss *= scale;
  mae_loss *= scale;
}

void PerfMetrics::print(Metrics const *m) {
  std::string output = "[Metrics]";
  if (train_all == 0) {
    double current_time = Realm::Clock::current_time_in_microseconds();
    assert(current_time > start_time);
    double throughput =
        (double)train_all / ((current_time - start_time) * 1e-6);
    output =
        output + " throughput: " + std::to_string(throughput) + "samples/s";
  }
  if (m->measure_accuracy) {
    float accuracy = train_correct * 100.0f / train_all;
    output = output + " accuracy: " + std::to_string(accuracy) + "% (" +
             std::to_string(train_correct) + " / " + std::to_string(train_all) +
             ")";
  }
  if (m->measure_categorical_crossentropy) {
    float avg_cce_loss = cce_loss / train_all;
    output =
        output + " categorical_crossentropy: " + std::to_string(avg_cce_loss);
  }
  if (m->measure_sparse_categorical_crossentropy) {
    float avg_cce_loss = sparse_cce_loss / train_all;
    output = output + " sparse_categorical_crossentropy: " +
             std::to_string(avg_cce_loss);
  }
  if (m->measure_mean_squared_error) {
    output =
        output + " mean_squared_error: " + std::to_string(mse_loss / train_all);
  }
  if (m->measure_root_mean_squared_error) {
    output = output + " root_mean_squared_error: " +
             std::to_string(rmse_loss / train_all);
  }
  if (m->measure_mean_absolute_error) {
    output = output +
             " mean_absolute_error: " + std::to_string(mae_loss / train_all);
  }
  fprintf(stderr, "%s\n", output.c_str());
}

}; // namespace FlexFlow
