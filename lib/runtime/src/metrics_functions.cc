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

#include "metrics_functions.h"
#include "kernels/metrics_kernels.h"
#include "runtime/profiling.h"
#include "runtime/task_spec/typed_task_invocation.h"
#include "task_spec/index_task_invocation.h"
#include "task_spec/task_argument_accessor.h"
#include "tasks.h"

namespace FlexFlow {

LegionRuntime::Logger::Category log_metrics("metrics");

MetricsAttrs::MetricsAttrs(LossFunction _loss_type,
                           std::vector<Metric> const &metrics)
    : loss_type(_loss_type), measure_accuracy(false),
      measure_categorical_crossentropy(false),
      measure_sparse_categorical_crossentropy(false),
      measure_mean_squared_error(false), measure_root_mean_squared_error(false),
      measure_mean_absolute_error(false) {
  for (Metric const &m : metrics) {
    switch (m) {
      case Metric::ACCURACY:
        measure_accuracy = true;
        continue;
      case Metric::CATEGORICAL_CROSSENTROPY:
        measure_categorical_crossentropy = true;
        continue;
      case Metric::SPARSE_CATEGORICAL_CROSSENTROPY:
        measure_sparse_categorical_crossentropy = true;
        continue;
      case Metric::MEAN_SQUARED_ERROR:
        measure_mean_squared_error = true;
        continue;
      case Metric::ROOT_MEAN_SQUARED_ERROR:
        measure_root_mean_squared_error = true;
        continue;
      case Metric::MEAN_ABSOLUTE_ERROR:
        measure_mean_absolute_error = true;
        continue;
      default:
        throw mk_runtime_error("Unrecogonized metrics type {}", m);
    }
  }
}

enum Slots {
  LOGIT,
  LABEL,
  METRICS_STRUCT,
  ALL_METRICS,
  ONE_METRICS,
  PROFILING_SETTINGS
};

TypedIndexTaskInvocation<PerfMetrics>
    compute_metrics(MetricsAttrs const &metrics,
                    parallel_tensor_guid_t const &logit,
                    parallel_tensor_guid_t const &label) {
  IndexTaskBinding binding(LOGIT);
  binding.bind(LOGIT, logit);
  binding.bind(LABEL, label);
  binding.bind_arg(METRICS_STRUCT, metrics);
  binding.bind_arg(PROFILING_SETTINGS, profiling_settings());

  return ensure_return_type<PerfMetrics>({METRICS_COMP_TASK_ID, binding});
}

TypedStandardTaskInvocation<PerfMetrics>
    update_metrics(MetricsAttrs const &metrics,
                   StandardTypedTaskArg<PerfMetrics> const &all_metrics,
                   TypedIndexTaskInvocation<PerfMetrics> const &one_metrics) {
  StandardTaskBinding binding;
  binding.bind_arg(METRICS_STRUCT, metrics);
  binding.bind_arg(ALL_METRICS, all_metrics);
  binding.bind_arg(ONE_METRICS, one_metrics);
  binding.bind_arg(PROFILING_SETTINGS, profiling_settings());

  return ensure_return_type<PerfMetrics>({UPDATE_METRICS_TASK_ID, binding});
}

TypedStandardTaskInvocation<PerfMetrics> compute_and_update_metrics(
    MetricsAttrs const &metrics,
    StandardTypedTaskArg<PerfMetrics> const &all_metrics,
    parallel_tensor_guid_t const &logit,
    parallel_tensor_guid_t const &label) {
  return update_metrics(
      metrics, all_metrics, compute_metrics(metrics, logit, label));
}

TaskInvocation reset_metrics(MetricsAttrs const &metrics) {
  StandardTaskBinding binding;

  binding.bind_arg(METRICS_STRUCT, metrics);

  return {UPDATE_METRICS_TASK_ID, binding};
}

//   // Use the same parallel strategy as the owner of logit
//   Context ctx = model->config.legion_config.lg_ctx;
//   Runtime *runtime = model->config.legion_config.lg_hlr;
//   Domain part_domain = runtime->get_index_space_domain(ctx,
//   logit->parallel_is); Domain logit_domain =
//   runtime->get_index_partition_color_space(
//       ctx, logit->part.get_index_partition());
//   Domain label_domain = runtime->get_index_partition_color_space(
//       ctx, label->part.get_index_partition());
//   if ((logit_domain != part_domain) || (label_domain != part_domain)) {
//     fprintf(stderr,
//             "Encounter inconsistency in parallelizing loss computation\n");
//     assert(false);
//   }
//   ArgumentMap argmap;
//   IndexLauncher launcher(METRICS_COMP_TASK_ID,
//                          logit->parallel_is,
//                          TaskArgument(this, sizeof(Metrics)),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          get_std_hash(logit->machine_view));
//   launcher.add_region_requirement(RegionRequirement(
//       logit->part, 0 /*projection id*/, Permissions::RO, EXCLUSIVE,
//       logit->region));
//   launcher.add_field(0, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(
//       label->part, 0 /*projection id*/, Permissions::RO, EXCLUSIVE,
//       label->region));
//   launcher.add_field(1, FID_DATA);
//   FutureMap new_metrics = runtime->execute_index_space(ctx, launcher);
//   // Update metrics
//   TaskLauncher metrics_task(UPDATE_METRICS_TASK_ID,
//                             TaskArgument(this, sizeof(Metrics)));
//   metrics_task.add_future(model->current_metrics);
//   for (Domain::DomainPointIterator it(part_domain); it; it++) {
//     metrics_task.add_future(new_metrics[*it]);
//   }
//   model->current_metrics = runtime->execute_task(ctx, metrics_task);
// }

static PerfMetrics make_empty_metrics() {
  return {static_cast<double>(Realm::Clock::current_time_in_microseconds())};
}

static PerfMetrics
    compute_metrics_task(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  auto me = acc.get_argument<MetricsAttrs>(METRICS_STRUCT);
  auto logit = acc.get_tensor<Permissions::RO>(LOGIT);
  auto label = acc.get_tensor<Permissions::RO>(LABEL);
  auto profiling_settings =
      acc.get_argument<ProfilingSettings>(PROFILING_SETTINGS);

  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  PerfMetrics perf_zc = make_empty_metrics();

  if (me.loss_type == LossFunction::SPARSE_CATEGORICAL_CROSSENTROPY) {
    // TensorAccessorR<float, NDIM> acc_logit(
    //     regions[0], task->regions[0], FID_DATA, ctx, runtime);
    // TensorAccessorR<int, NDIM> acc_label(
    //     regions[1], task->regions[1], FID_DATA, ctx, runtime);

    // assume that the leading dim is replica dim
    assert(logit.shape.at(logit.shape.last_idx()) == 1);
    int num_effective_samples = label.shape.get_volume();
    int num_classes = logit.shape.at(legion_dim_t(0));
    assert(num_effective_samples * num_classes == logit.shape.get_volume());
    assert(label.shape.sub_shape(legion_dim_t(1), nullopt) ==
           logit.shape.sub_shape(legion_dim_t(1), nullopt));
    assert(label.shape.at(legion_dim_t(0)) == 1);
    // Cannot measure categorical_crossentropy w/ sparse labels
    // Use measure_sparse_categorical_crossentropy instead
    assert(!me.measure_categorical_crossentropy);
    profile(update_metrics_sparse_label_kernel,
            profiling_settings,
            "[Compute Metrics] running_time = %.2lfms\n",
            me,
            get_float_ptr(logit),
            get_int32_ptr(label),
            num_effective_samples,
            num_classes,
            perf_zc);
  } else {
    // other loss require label and logit have identical shape
    assert(logit.shape == label.shape);
    // assume that the leading dim is replica dim
    assert(logit.shape.at(logit.shape.last_idx()) == 1);
    int num_samples = logit.shape.at(logit.shape.neg_idx(1));
    int num_classes = logit.shape.get_volume() / num_samples;
    // Use CUDA_NUM_THREADS may result in out of resources so we set
    // #threads=256
    profile(update_metrics_label_kernel,
            profiling_settings,
            "[Compute Mtrics] running_time = %.2lfms\n",
            me,
            get_float_ptr(logit),
            get_float_ptr(label),
            num_samples,
            num_classes,
            perf_zc);
  }
  return perf_zc;
}

static PerfMetrics
    update_metrics_task(Legion::Task const *task,
                        std::vector<Legion::PhysicalRegion> const &regions,
                        Legion::Context ctx,
                        Legion::Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  auto m = acc.get_argument<MetricsAttrs>(METRICS_STRUCT);
  auto maybe_all_metrics = acc.get_optional_argument<PerfMetrics>(ALL_METRICS);
  auto one_metrics = acc.get_variadic_argument<PerfMetrics>(ONE_METRICS);
  auto profiling_settings =
      acc.get_argument<EnableProfiling>(PROFILING_SETTINGS);

  if (!maybe_all_metrics.has_value()) {
    assert(one_metrics.empty());
    return make_empty_metrics();
  }

  assert(!one_metrics.empty());
  PerfMetrics all_metrics = maybe_all_metrics.value();
  for (PerfMetrics const &pm : one_metrics) {
    all_metrics = update(all_metrics, pm);
  }
  log_metrics.print() << fmt::to_string(all_metrics);
  return all_metrics;
}

template <>
void register_task<METRICS_COMP_TASK_ID>() {
  TaskSignature sig;
  sig.add_slot(LOGIT, {SlotType::TENSOR, Permissions::RO});
  sig.add_slot(LABEL, {SlotType::TENSOR, Permissions::RO});
  sig.add_arg_slot<ProfilingSettings>(PROFILING_SETTINGS);
  sig.add_arg_slot<MetricsAttrs>(METRICS_STRUCT);
  sig.add_return_value<PerfMetrics>();

  register_task(
      METRICS_COMP_TASK_ID, "Metrics Compute", sig, compute_metrics_task);
}

template <>
void register_task<UPDATE_METRICS_TASK_ID>() {
  TaskSignature sig;
  sig.add_arg_slot<MetricsAttrs>(METRICS_STRUCT);
  sig.add_arg_slot<PerfMetrics>(ALL_METRICS);
  sig.add_arg_slot<ProfilingSettings>(PROFILING_SETTINGS);
  sig.add_variadic_arg_slot<PerfMetrics>(ONE_METRICS);
  sig.add_return_value<PerfMetrics>();

  register_task(
      UPDATE_METRICS_TASK_ID, "Update Metrics", sig, update_metrics_task);
}

} // namespace FlexFlow
