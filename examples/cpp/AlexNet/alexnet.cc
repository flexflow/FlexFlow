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

#include "alexnet.h"
#include <fstream>
#include <sstream>
#include <string>
using namespace Legion;
using FlexFlow::FFConfig;
using FlexFlow::FFModel;
using FlexFlow::Optimizer;
using FlexFlow::ParallelDim;
using FlexFlow::ParallelTensor;
using FlexFlow::SGDOptimizer;
using FlexFlow::Tensor;

Legion::Logger log_app("AlexNet");

void parse_input_args(char **argv, int argc, AlexNetConfig &config) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--dataset")) {
      std::strcpy(config.dataset_path, argv[++i]);
      continue;
    }
  }
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffConfig;
  AlexNetConfig alexnetConfig;
  {
    InputArgs const &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, alexnetConfig);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
                  ffConfig.batchSize,
                  ffConfig.workersPerNode,
                  ffConfig.numNodes);
  }
  FFModel ff(ffConfig);

  Tensor input;
  {
    int const dims[] = {ffConfig.batchSize, 3, 229, 229};
    input = ff.create_tensor<4>(dims, DT_FLOAT);
  }
  // Tensor label;
  //{
  //   const int dims[] = {ffConfig.batchSize, 1};
  //   label = ff.create_tensor<2>(dims, DT_INT32);
  // }
  //  Add layers
  Tensor t = input, ts[2];
  t = ff.conv2d(input, 64, 11, 11, 4, 4, 2, 2, AC_MODE_RELU);
  // ts[1] = ff.conv2d("conv1", input, 64, 11, 11, 4, 4, 2, 2);
  // t = ff.concat("concat", 2, ts, 1/*axis*/);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  t = ff.conv2d(t, 192, 5, 5, 1, 1, 2, 2, AC_MODE_RELU);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  t = ff.conv2d(t, 384, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t = ff.conv2d(t, 256, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t = ff.conv2d(t, 256, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  t = ff.flat(t);
  t = ff.dense(t, 4096, AC_MODE_RELU /*relu*/);
  t = ff.dense(t, 4096, AC_MODE_RELU /*relu*/);
  t = ff.dense(t, 10);
  t = ff.softmax(t);
  Optimizer *optimizer = new SGDOptimizer(&ff, 0.001f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);
  // Data Loader
  ParallelTensor input_pt, label_pt;
  ff.get_parallel_tensor_from_tensor(input, input_pt);
  ff.get_parallel_tensor_from_tensor(ff.label_tensor, label_pt);
  DataLoader data_loader(ff, &alexnetConfig, input_pt, label_pt);
  ff.init_operators();
  // Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    data_loader.reset();
    ff.reset_metrics();
    int iterations = data_loader.num_samples / ffConfig.batchSize;

    for (int iter = 0; iter < iterations; iter++) {
      if (std::strlen(alexnetConfig.dataset_path) == 0) {
        // Only load data once for random input
        if (iter == 0 && epoch == 0) {
          data_loader.next_batch(ff);
        }
      } else {
        data_loader.next_batch(ff);
      }
      // runtime->begin_trace(ctx, 111 /*trace_id*/);
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
      // runtime->end_trace(ctx, 111 /*trace_id*/);
    }
  }
  // End timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double run_time = 1e-6 * (ts_end - ts_start);
  printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n",
         run_time,
         data_loader.num_samples * ffConfig.epochs / run_time);
}

size_t get_file_size(std::string const &filename) {
  streampos begin, end;
  ifstream file(filename.c_str(), ios::binary);
  begin = file.tellg();
  file.seekg(0, ios::end);
  end = file.tellg();
  file.close();
  size_t filesize = end - begin;
  return filesize;
}

DataLoader::DataLoader(FFModel &ff,
                       AlexNetConfig const *alexnet,
                       ParallelTensor input,
                       ParallelTensor label) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  num_samples = 0;
  if (std::strlen(alexnet->dataset_path) == 0) {
    log_app.print("Use random dataset...");
    num_samples = 1024 * ff.config.workersPerNode * ff.config.numNodes;
    log_app.print("Number of random samples = %d\n", num_samples);
  } else {
    log_app.print("Start loading dataset from %s", alexnet->dataset_path);
    size_t filesize = get_file_size(alexnet->dataset_path);
    assert(filesize % 3073 == 0);
    num_samples = filesize / 3073;
  }

  // Create full input
  {
    assert(input->num_dims == 5);
    batch_input = input;
    ParallelDim dims[5];
    for (int i = 0; i < 5; i++) {
      dims[i].size = input->dims[i].size;
      dims[i].degree = 1;
      dims[i].parallel_idx = -1;
      dims[i].is_replica_dim = input->dims[i].is_replica_dim;
      // Assume only the first dim can be the replica dim
      assert(i == 4 || (!dims[i].is_replica_dim));
    }
    dims[3].size = num_samples;
    // int const dims[] = {
    //     num_samples, input->dims[2], input->dims[1], input->dims[0]};
    full_input = ff.create_parallel_tensor_legion_ordering(5, dims, DT_FLOAT);
    ff.map_tensor(full_input, NULL /*parallel_op*/);
  }
  // Create full label
  {
    assert(label->num_dims == 3);
    batch_label = label;
    ParallelDim dims[3];
    for (int i = 0; i < 3; i++) {
      dims[i].size = label->dims[i].size;
      dims[i].degree = 1;
      dims[i].parallel_idx = -1;
      // Assume only the first dim can be the replica dim
      assert(i == 2 || (!dims[i].is_replica_dim));
    }
    dims[1].size = num_samples;
    // int const dims[] = {num_samples, label->dims[0]};
    full_label = ff.create_parallel_tensor_legion_ordering(3, dims, DT_INT32);
    ff.map_tensor(full_label, NULL);
  }
  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  TaskLauncher launcher(FlexFlow::CUSTOM_CPU_TASK_ID_1,
                        TaskArgument(alexnet, sizeof(AlexNetConfig)));
  // regions[0]: full_input
  launcher.add_region_requirement(RegionRequirement(full_input->region,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    full_input->region,
                                                    MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_label
  launcher.add_region_requirement(RegionRequirement(full_label->region,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    full_label->region,
                                                    MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  runtime->execute_task(ctx, launcher);
  reset();
  next_batch(ff);
}

__inline__ int calc_offset(int c, int y, int x, int yscale, int xscale) {
  return (c * yscale * xscale + y * xscale + x);
}

void nearest_neigh(unsigned char *image,
                   unsigned char *buffer,
                   int height,
                   int width,
                   int orig_height,
                   int orig_width,
                   float height_scale,
                   float width_scale) {
  for (int y = 0; y < height; y++) {
    int y0 =
        std::min(static_cast<int>(roundf(y * height_scale)), orig_height - 1);
    for (int x = 0; x < width; x++) {
      int x0 =
          std::min(static_cast<int>(roundf(x * width_scale)), orig_width - 1);
      for (int c = 0; c < 3; c++) {
        int origOffset = calc_offset(y0, x0, c, orig_width, 3);
        int offset = calc_offset(c, y, x, height, width);
        image[offset] = buffer[origOffset];
      }
    }
  }
}

void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  AlexNetConfig const *alexnet = (AlexNetConfig *)task->args;
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  const FlexFlow::AccessorWO<float, 5> acc_input(regions[0], FID_DATA);
  const FlexFlow::AccessorWO<int, 3> acc_label(regions[1], FID_DATA);
  Rect<5> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  Rect<3> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  float *input_ptr = acc_input.ptr(rect_input.lo);
  int *label_ptr = acc_label.ptr(rect_label.lo);
  int num_samples = rect_label.hi[1] - rect_label.lo[1] + 1;
  assert(rect_input.hi[3] - rect_input.lo[3] + 1 == num_samples);
  if (std::strlen(alexnet->dataset_path) == 0) {
    log_app.print("Start generating random input samples");
    for (size_t i = 0; i < rect_label.volume(); i++) {
      label_ptr[i] = std::rand() % 10;
    }
    return;
  }
  log_app.print(
      "Start loading %d samples from %s\n", num_samples, alexnet->dataset_path);
  int height = rect_input.hi[1] - rect_input.lo[1] + 1;
  int width = rect_input.hi[0] - rect_input.lo[0] + 1;
  int origHeight = 32;
  int origWidth = 32;
  float heightScale = static_cast<float>(origHeight) / height;
  float widthScale = static_cast<float>(origWidth) / width;
  FILE *file = fopen(alexnet->dataset_path, "rb");
  unsigned char *buffer = (unsigned char *)malloc(3073);
  unsigned char *image = (unsigned char *)malloc(3 * height * width);
  for (off_t i = 0; i < num_samples; i++) {
    size_t ret = fread(buffer, sizeof(unsigned char), 3073, file);
    assert(ret = 3073);
    if ((i + 1) % 1000 == 0) {
      log_app.print("Loaded %ld samples", i + 1);
    }
    label_ptr[i] = buffer[0];
    nearest_neigh(image,
                  buffer + 1,
                  height,
                  width,
                  origHeight,
                  origWidth,
                  heightScale,
                  widthScale);
    off_t input_offset = i * 3 * height * width;
    off_t image_offset = 0;
    for (off_t h = 0; h < 3 * height * width; h++) {
      input_ptr[input_offset++] =
          static_cast<float>(image[image_offset++]) / 255;
    }
  }
  log_app.print("Finish loading %d samples from %s\n",
                num_samples,
                alexnet->dataset_path);
  fclose(file);
}

void DataLoader::next_batch(FFModel &ff) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Load input
  {
    Domain domain =
        runtime->get_index_space_domain(ctx, batch_input->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % batch_input->dims[3].size == 0);
      meta.num_samples = batch_input->dims[3].size;
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(FlexFlow::CUSTOM_GPU_TASK_ID_1,
                           batch_input->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_input->machine_view.hash());
    launcher.add_region_requirement(RegionRequirement(full_input->region,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      full_input->region,
                                                      MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(batch_input->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      batch_input->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load label
  {
    Domain domain =
        runtime->get_index_space_domain(ctx, batch_label->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % batch_label->dims[1].size == 0);
      meta.num_samples = batch_label->dims[1].size;
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(FlexFlow::CUSTOM_GPU_TASK_ID_2,
                           batch_label->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_label->machine_view.hash());
    launcher.add_region_requirement(RegionRequirement(full_label->region,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      full_label->region,
                                                      MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(batch_label->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      batch_label->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  next_index += ff.config.batchSize;
}

void DataLoader::reset() {
  next_index = 0;
}

void FlexFlow::register_custom_tasks() {
  // Load entire dataset
  {
    TaskVariantRegistrar registrar(FlexFlow::CUSTOM_CPU_TASK_ID_1,
                                   "Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_entire_dataset>(
        registrar, "Load Entire Dataset Task");
  }
  // Load input
  {
    TaskVariantRegistrar registrar(FlexFlow::CUSTOM_GPU_TASK_ID_1,
                                   "Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Input Task");
  }
  // Load label
  {
    TaskVariantRegistrar registrar(FlexFlow::CUSTOM_GPU_TASK_ID_2,
                                   "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(
        registrar, "Load Label Task");
  }
}
