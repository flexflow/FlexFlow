/* Copyright 2020 Stanford
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

#include "model.h"
#include "cuda_helper.h"

Metrics::Metrics(const std::vector<std::string>& metrics)
: measure_accuracy(false),
  measure_categorical_crossentropy(false),
  measure_sparse_categorical_crossentropy(false),
  measure_mean_squared_error(false),
  measure_root_mean_sequarted_error(false),
  measure_mean_absolute_error(false)
{
  for (size_t i = 0; i < metrics.length(); i++) {
    switch (metrics[i]) {
      case "accuracy":
        measure_accuracy = true;
        continue;
      case "categorical_crossentropy":
        measure_categorical_crossentropy = true;
        continue;
      case "sparse_categorical_crossentropy":
        measure_sparse_categorical_crossentropy = true;
        continue;
      case "mean_squared_error":
        measure_mean_squared_error = true;
        continue;
      case "root_mean_sequarted_error":
        measure_root_mean_sequarted_error = true;
        continue;
      case "mean_absolute_error":
        measure_mean_absolute_error = true;
        continue;
      default:
        fprintf(stderr, "Unrecogonized Metrics: %s.\n", metrics[i].c_str());
        assert(false);
    }
  }
}

__global__
void update_metrics_sparse_label_kernel(
    const float* logits,
    const int* labels,
    PerfMetrics* perf
    const Metrics metrics,
    int num_samples,
    int num_classes)
{
  CUDA_KERNEL_LOOP(b, num_samples)
  {
    if (metrics.measure_accuracy) {
      float max_val = -1.0f;
      int my_label = -1;
      for (int i = 0; i < num_classes; i++) {
        float my_logit = logits[b*num_classes+i];
        if (my_logit > max_val) {
          max_val = my_logit;
          my_label = i;
        }
      }
      assert(my_label >= 0);
      atomicAdd(&(perf->train_all), 1);
      if (labels[b] == my_label)
        atomicAdd(&(perf->train_correct), 1);
    }
    if (metrics.measure_sparse_categorical_crossentropy) {
      float my_logit = max(logits[b*num_labels+labels[b]], LOG_MIN_VALUE);
      atomicAdd(&(perf->sparse_cce_loss), -log(my_logit));
    }
    if (metrics.measure_mean_squared_error
    || metrics.measure_root_mean_squarted_error
    || metrics.measure_mean_absolute_error)
    {
      float mse = 0.0f, mae = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        float my_logit = logits[b*num_classes+i];
        float my_label = (labels[b] == i) ? 1.0f : 0.0f;
        mse += (my_logit - my_label) * (my_logit - my_label);
        mae += abs(my_logit - my_label);
      }
      if (metrics.measure_mean_squared_error)
        atomicAdd(&(perf->mse_loss), mse);
      if (metrics.measure_root_mean_squared_error)
        atomicAdd(&(perf->rmse_loss), sqrt(mse));
      if (metrics.measure_mean_absolute_error)
        atomicAdd(&(perf->mae_loss), mae);
    }
  }
}

__global__
void update_metrics_label_kernel(
    const float* logits,
    const float* labels,
    PerfMetrics* perf
    const Metrics metrics,
    int num_samples,
    int num_classes)
{
  CUDA_KERNEL_LOOP(b, num_samples)
  {
    if (metrics.measure_accuracy) {
      float max_val = -1.0f;
      int my_label = -1, true_label = -1;
      for (int i = 0; i < num_classes; i++) {
        float my_logit = logits[b*num_classes+i];
        float my_label = labels[b*num_classes+i];
        if (my_logit > max_val) {
          max_val = my_logit;
          my_label = i;
        }
        if (my_label > 0.9f) {
          assert(true_label == -1);
          true_label = i;
        }
      }
      assert(my_label >= 0);
      assert(true_label >= 0);
      atomicAdd(&(perf->train_all), 1);
      if (true_label == my_label)
        atomicAdd(&(perf->train_correct), 1);
    }
    if (metrics.measure_categorical_crossentropy) {
      float cce = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        if (labels[b*num_classes+i] > 0.0f) {
          float my_logit = max(logits[b*num_labels+labels[b]], LOG_MIN_VALUE);
          cce += labels[b*num_classes+i] * -log(my_logit);
        }
      }
      atomicAdd(&(perf->sparse_cce_loss), cce);
    }
    if (metrics.measure_mean_squared_error
    || metrics.measure_root_mean_squarted_error
    || metrics.measure_mean_absolute_error)
    {
      float mse = 0.0f, mae = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        float my_logit = logits[b*num_classes+i];
        float my_label = labels[b*num_classes+i];
        mse += (my_logit - my_label) * (my_logit - my_label);
        mae += abs(my_logit - my_label);
      }
      if (metrics.measure_mean_squared_error)
        atomicAdd(&(perf->mse_loss), mse);
      if (metrics.measure_root_mean_squared_error)
        atomicAdd(&(perf->rmse_loss), sqrt(mse));
      if (metrics.measure_mean_absolute_error)
        atomicAdd(&(perf->mae_loss), mae);
    }
  }
}

__host__
PerfMetrics Metrics::compute_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Metrics* me = (Metrics*) task->args;
  PerfMetrics* perf;
  PerfMetrics perf_zc;
  checkCUDA(cudaMalloc(&perf, sizeof(PerfMetrics)));
  checkCUDA(cudaMemcpy(perf, &perf_zc, sizeof(PerfMetrics), cudaMemcpyHostToDevice));

  if (me->loss_type == Loss::SPARSE_CATEGORICAL_CROSSENTROPY) {
    TensorAccessorR<float, 2> acc_logit(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    TensorAccessorR<int, 2> acc_label(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    int num_samples = acc_logit.rect.hi[1] - acc_logit.rect.lo[1] + 1;
    int num_classes = acc_logit.rect.hi[0] - acc_logit.rect.lo[0] + 1;a
    assert(acc_label.rect.hi[1] == acc_logit.rect.hi[1]);
    assert(acc_label.rect.lo[1] == acc_logit.rect.lo[1]);
    assert(acc_label.rect.lo[0] == acc_label.rect.hi[0]);
    // Cannot measure categorical_crossentropy w/ sparse labels
    // Use measure_sparse_categorical_crossentropy instead
    assert(!me->measure_categorical_crossentropy);
    update_metrics_sparse_label_kernel<<<GET_BLOCKS(num_samples), CUDA_NUM_THREADS>>>(
        acc_logit.ptr, acc_label.ptr, perf, *me, num_samples, num_classes);
  } else {
    TensorAccessorR<float, 2> acc_logit(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    TensorAccessorR<float, 2> acc_label(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    // other loss require label and logit have identical shape
    assert(acc_logit.rect == acc_label.rect);
    update_metrics_label_kernel<<<GET_BLOCKS(num_samples), CUDA_NUM_THREADS>>>(
      acc_logit.ptr, acc_label.ptr, perf, *me, num_samples, num_classes);
  }
  checkCUDA(cudaMemcpy(&perf_zc, perf, sizeof(PerfMetrics), cudaMemcpyDeviceToHost));
  checkCUDA(cudaFree(perf));
  return perf_zc;
}

void Metrics::compute()
{
  // Use the same parallel strategy as the owner of logit
  std::string pcname = logit->owner_op->name;
  IndexSpaceT<2> task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);
  Rect<2> logit_rect = runtime->get_index_partition_color_space(
      ctx, logit->part.get_index_partition());
  Rect<2> label_rect = runtime->get_index_partition_color_space(
      ctx, label->part.get_index_partition());
  if((logit_rect != part_rect) || (label_rect != part_rect)) {
    fprintf(stderr, "Encounter inconsistency in parallelizing loss computation");
    assert(false);
  }
  ArgumentMap argmap;
  IndexLauncher launcher(LOSS_FUNC_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Metrics)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(pcname));
  launcher.add_region_requirement(
      RegionRequirement(logit->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, logit->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(label->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, label->region));
  launcher.add_field(1, FID_DATA);
  FutureMap new_metrics = runtime->execute_index_space(ctx, launcher);
  // Update metrics
  TaskLauncher metrics_task(UPDATE_METRICS_TASK_ID, TaskArgument(NULL, 0));
  metrics_task.add_future(ff.current_metrics);
  for (PointInRectIterator<2> it(part_rect); it(); it++) {
    metrics_task.add_future(new_metrics[*it]);
  }
  ((FFModel*)(&ff))->current_metrics = runtime->execute_task(ctx, metrics_task);
}
