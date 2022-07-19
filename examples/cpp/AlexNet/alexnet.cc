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

#include "alexnet.h"
#include <sstream>
#include <fstream>
#include <string>
using namespace Legion;
using namespace FlexFlow;
using FlexFlow::FFModel;
using FlexFlow::Tensor;
using FlexFlow::FFConfig;
using FlexFlow::Optimizer;
using FlexFlow::SGDOptimizer;


LegionRuntime::Logger::Category log_app("AlexNet");

void parse_input_args(char **argv, int argc, AlexNetConfig& config)
{
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--dataset")) {
      std::strcpy(config.dataset_path, argv[++i]);
      continue;
    }
  }
}

void FlexFlow::top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
  FFConfig ffConfig;
  AlexNetConfig alexnetConfig;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, alexnetConfig);
    // log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
    //     ffConfig.batchSize, ffConfig.workersPerNode, ffConfig.numNodes);
    log_app.print("ubatchSize(%d) workersPerNodes(%d) numNodes(%d) dataset(%s)",
        ffConfig.ubatchUnit, ffConfig.workersPerNode, ffConfig.numNodes, alexnetConfig.dataset_path);
  }
  FFModel ff(ffConfig);

  Tensor input;
  {
    const int dims[] = {ffConfig.ubatchUnit, 3, 229, 229};
    input = ff.create_tensor<4>(dims, DT_FLOAT);
  }
  //Tensor label;
  //{
  //  const int dims[] = {ffConfig.batchSize, 1};
  //  label = ff.create_tensor<2>(dims, DT_INT32);
  //}
  // Add layers
  Tensor t = input, ts[2];
  t = ff.conv2d(input, 64, 11, 11, 4, 4, 2, 2, AC_MODE_RELU);
  //ts[1] = ff.conv2d("conv1", input, 64, 11, 11, 4, 4, 2, 2);
  //t = ff.concat("concat", 2, ts, 1/*axis*/);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  t = ff.conv2d(t, 192, 5, 5, 1, 1, 2, 2, AC_MODE_RELU);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  t = ff.conv2d(t, 384, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t = ff.conv2d(t, 256, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t = ff.conv2d(t, 256, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  t = ff.flat(t);
  t = ff.dense(t, 4096, AC_MODE_RELU/*relu*/);
  t = ff.dense(t, 4096, AC_MODE_RELU/*relu*/);
  t = ff.dense(t, 10);
  t = ff.softmax(t);
  Optimizer* optimizer = new SGDOptimizer(&ff, 0.001f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);
  // Data Loader TODO: change
  log_app.print("DEBUG: finish model compilation");
  DataLoader data_loader(ff, &alexnetConfig, input, ff.label_tensor);
  ff.init_operators();
  log_app.print("DEBUG: finish op init");
  //Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();
  printf("Start Traning.....\n");
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    printf("Current Epoch: %d\n", epoch);
    data_loader.reset();
    ff.reset_metrics();
    int iterations = data_loader.num_samples / ffConfig.batchSize;

    for (int iter = 0; iter < iterations; iter++) {
      log_app.print("hereeee iter %d",iter);
      runtime->begin_trace(ctx, 111/*trace_id*/);
      for (int iter_inner =0; iter_inner < ff.iter_perbatch; iter_inner++){
        if (std::strlen(alexnetConfig.dataset_path) == 0) {
          // Only load data once for random input
          if (iter == 0 && epoch == 0)
            data_loader.next_batch(ff);
        } else {
          //shicao pipeline
          for (int i=0; i < data_loader.batch_input->parallel_tensor->owner_op->nFnB; i++){
            data_loader.next_input_ubatch(ff);
          }

          for (int i=0; i < ff.get_final_operator()->nFnB; i++){
            data_loader.next_label_ubatch(ff);
          }
        }
        log_app.print("DEBUG: forward...");
        ff.forward();
	log_app.print("DEBUG: backward...");
        ff.backward();
      }
      log_app.print("DEBUG:update weight");
      ff.update();
      log_app.print("DEBUG:zero gradients");
      ff.zero_gradients();
      log_app.print("DEBUG:zero gradients finish");
      runtime->end_trace(ctx, 111/*trace_id*/);
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
  printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n", run_time,
         data_loader.num_samples * ffConfig.epochs / run_time);
}

size_t get_file_size(const std::string& filename)
{
  streampos begin,end;
  ifstream file(filename.c_str(), ios::binary);
  begin = file.tellg();
  file.seekg (0, ios::end);
  end = file.tellg();
  file.close();
  size_t filesize = end - begin;
  return filesize;
}

DataLoader::DataLoader(FFModel& ff,
                       const AlexNetConfig* alexnet,
                       Tensor input, Tensor label)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  num_samples = 0;
  if (std::strlen(alexnet->dataset_path) == 0) {
    log_app.print("Use random dataset...");
    num_samples = 1024 * ff.config.workersPerNode * ff.config.numNodes;
    log_app.print("Number of random samples = %d\n", num_samples);
  } else {
    log_app.print("Start loading dataset from %s", alexnet->dataset_path);
    size_t filesize = get_file_size(alexnet->dataset_path);
    assert(filesize % 3073 == 0);
    num_samples = filesize / 3073 / 100;
  }
  // Create full input
  {
    batch_input = input;
    const int dims[] = {num_samples, input->dims[2], input->dims[1], input->dims[0]};
    ParallelDim pdims[4];
    pdims[0].size = num_samples;
    pdims[0].parallel_idx = -1;
    pdims[1].size = input->dims[2];
    pdims[2].size = input->dims[1];
    pdims[3].size = input->dims[0];
    pdims[1].parallel_idx = -1;
    pdims[2].parallel_idx = -1;
    pdims[3].parallel_idx = -1;
    pdims[0].degree = 1;
    pdims[1].degree = 1;
    pdims[2].degree = 1;
    pdims[3].degree = 1;
    full_input = ff.create_tensor<4>(dims, DT_FLOAT);
    log_app.print("Created full input tensor...");
    full_input->parallel_tensor = ff.create_parallel_tensor<4>(pdims, DT_FLOAT);
    log_app.print("Created full input parallel tensor...");
    ff.map_tensor(full_input->parallel_tensor, NULL);
    log_app.print("Mapped full input parallel tensor...");
  }
  // Create full label
  {
    batch_label = label;
    const int dims[] = {num_samples, label->dims[0]};
    ParallelDim pdims[2];
    pdims[0].size = num_samples;
    pdims[1].size = label->dims[0];
    pdims[0].parallel_idx = -1;
    pdims[1].parallel_idx = -1;
    pdims[0].degree = 1;
    pdims[1].degree = 1;
    log_app.print("Full label tensor(%d, %d)", label->dims[0], num_samples);
    full_label = ff.create_tensor<2>(dims, DT_INT32);
    full_label->parallel_tensor = ff.create_parallel_tensor<2>(pdims, DT_INT32);
    log_app.print("Created full label parallel tensor...");
    ff.map_tensor(full_label->parallel_tensor, NULL);
  }
  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  assert(full_input->parallel_tensor != nullptr);
  assert(full_input->parallel_tensor->region != LogicalRegion::NO_REGION);
  assert(full_label->parallel_tensor->region != LogicalRegion::NO_REGION);
  TaskLauncher launcher(FlexFlow::CUSTOM_CPU_TASK_ID_1,
      TaskArgument(alexnet, sizeof(AlexNetConfig)));
  // regions[0]: full_input
  launcher.add_region_requirement(
      RegionRequirement(full_input->parallel_tensor->region, WRITE_ONLY,
                        EXCLUSIVE, full_input->parallel_tensor->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label->parallel_tensor->region, WRITE_ONLY,
                        EXCLUSIVE, full_label->parallel_tensor->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  runtime->execute_task(ctx, launcher);
  reset();
  log_app.print("Reset sample idx");
  next_input_ubatch(ff);
  next_label_ubatch(ff);
}

__inline__
int calc_offset(int c, int y, int x, int yscale, int xscale)
{
  return (c * yscale * xscale + y * xscale + x);
}

void nearest_neigh(unsigned char* image,
                   unsigned char* buffer,
                   int height, int width,
                   int orig_height, int orig_width,
                   float height_scale, float width_scale)
{
  for (int y = 0; y < height; y++) {
    int y0 = std::min(static_cast<int>(roundf(y * height_scale)), orig_height - 1);
    for (int x = 0; x < width; x++) {
      int x0 = std::min(static_cast<int>(roundf(x * width_scale)), orig_width - 1);
      for (int c = 0; c < 3; c++) {
        int origOffset = calc_offset(y0, x0, c, orig_width, 3);
        int offset = calc_offset(c, y, x, height, width);
        image[offset] = buffer[origOffset];
      }
    }
  }
}

void DataLoader::load_entire_dataset(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime* runtime)
{
  const AlexNetConfig* alexnet = (AlexNetConfig*)task->args;
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  const FlexFlow::AccessorWO<float, 4> acc_input(regions[0], FID_DATA);
  const FlexFlow::AccessorWO<int, 2> acc_label(regions[1], FID_DATA);
  Rect<4> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  Rect<2> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  float* input_ptr = acc_input.ptr(rect_input.lo);
  int* label_ptr = acc_label.ptr(rect_label.lo);
  int num_samples = rect_label.hi[1] - rect_label.lo[1] + 1;
  assert(rect_input.hi[3] - rect_input.lo[3] + 1 == num_samples);
  if (std::strlen(alexnet->dataset_path) == 0) {
    log_app.print("Start generating random input samples");
    for (size_t i = 0; i < rect_label.volume(); i++)
      label_ptr[i] = std::rand() % 10;
    return;
  }
  log_app.print("Start loading %d samples from %s\n",
      num_samples, alexnet->dataset_path);
  int height = rect_input.hi[1] - rect_input.lo[1] + 1;
  int width = rect_input.hi[0] - rect_input.lo[0] + 1;
  int origHeight = 32;
  int origWidth = 32;
  float heightScale = static_cast<float>(origHeight) / height;
  float widthScale = static_cast<float>(origWidth) / width;
  FILE* file = fopen(alexnet->dataset_path, "rb");
  unsigned char* buffer = (unsigned char*) malloc(3073);
  unsigned char* image = (unsigned char*) malloc(3 * height * width);
  for (off_t i = 0; i < num_samples; i++) {
    log_app.print("Loading sample (%ld)", i);
    size_t ret = fread(buffer, sizeof(unsigned char), 3073, file);
    assert(ret = 3073);
    if ((i+1) % 1000 == 0)
      log_app.print("Loaded %ld samples", i+1);
    label_ptr[i] = buffer[0];
    nearest_neigh(image, buffer + 1, height, width,
                  origHeight, origWidth, heightScale, widthScale);
    off_t input_offset = i * 3 * height * width;
    off_t image_offset = 0;
    for (off_t h = 0; h < 3*height*width; h++)
        input_ptr[input_offset++] = static_cast<float>(image[image_offset++]) / 255;
  }
  log_app.print("Finish loading %d samples from %s\n",
      num_samples, alexnet->dataset_path);
  fclose(file);
}

void DataLoader::next_batch(FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // Load input
  {
    IndexSpaceT<4> task_is = (IndexSpaceT<4>) batch_input->parallel_tensor->parallel_is;
    Rect<4> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<4> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[3] - rect.lo[3] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[3] - rect.lo[3] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(FlexFlow::CUSTOM_GPU_TASK_ID_1, task_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           batch_input->parallel_tensor->machine_view.hash());
    launcher.add_region_requirement(
        RegionRequirement(full_input->parallel_tensor->region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_input->parallel_tensor->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_input->parallel_tensor->part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_input->parallel_tensor->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load label
  {
    IndexSpaceT<2> task_is = IndexSpaceT<2>(batch_label->parallel_tensor->parallel_is);
    Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<2> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(FlexFlow::CUSTOM_GPU_TASK_ID_2, task_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           batch_label->parallel_tensor->machine_view.hash());
    launcher.add_region_requirement(
        RegionRequirement(full_label->parallel_tensor->region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_label->parallel_tensor->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_label->parallel_tensor->part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_label->parallel_tensor->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  next_index += ff.config.batchSize;
}

void DataLoader::next_input_ubatch(FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  int ubSize = batch_input->parallel_tensor->owner_op->ubSize;
  // Load input
  {
    // IndexSpaceT<4> task_is = (IndexSpaceT<4>) batch_input->parallel_tensor->parallel_is;
    Domain domain = runtime->get_index_space_domain(ctx, batch_input->parallel_tensor->parallel_is);
    ArgumentMap argmap;
    int idx = next_input_index;
    printf("Loading Input to buffer %d\n", input_idx);
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      int ndims = batch_input->parallel_tensor->num_dims;
      meta.num_samples = ubSize / batch_input->parallel_tensor->dims[ndims-2].degree;
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(FlexFlow::CUSTOM_GPU_TASK_ID_1, batch_input->parallel_tensor->parallel_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           batch_input->parallel_tensor->machine_view.hash());
    launcher.add_region_requirement(
        RegionRequirement(full_input->parallel_tensor->region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_input->parallel_tensor->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_input->parallel_tensor->in_pipepart[input_idx], 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_input->parallel_tensor->in_subregions[input_idx]));
    launcher.add_field(1, FID_DATA);
    input_idx = (input_idx + 1) % batch_input->parallel_tensor->pipe_num_part_in;
    runtime->execute_index_space(ctx, launcher);
  }
  next_input_index += ubSize;
}

void DataLoader::next_label_ubatch(FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  int ubSize = batch_label->parallel_tensor->pipe_buf_size / batch_label->parallel_tensor->pipe_num_part_out;
  // Load label
  {
    //IndexSpaceT<2> task_is = IndexSpaceT<2>(batch_label->parallel_tensor->parallel_is);
    Domain domain = runtime->get_index_space_domain(ctx, batch_label->parallel_tensor->parallel_is);
    ArgumentMap argmap;
    int idx = next_label_index;
    printf("Loading label to buffer %d\n", label_idx);
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      int ndims = batch_label->parallel_tensor->num_dims;
      meta.num_samples = ubSize/ batch_label->parallel_tensor->dims[ndims-2].degree;
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    assert(batch_label->parallel_tensor->out_subregions[label_idx] != LogicalRegion::NO_REGION);
    IndexLauncher launcher(FlexFlow::CUSTOM_GPU_TASK_ID_2, batch_label->parallel_tensor->parallel_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           batch_label->parallel_tensor->machine_view.hash());
    launcher.add_region_requirement(
        RegionRequirement(full_label->parallel_tensor->region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_label->parallel_tensor->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_label->parallel_tensor->out_pipepart[label_idx], 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_label->parallel_tensor->out_subregions[label_idx]));
    launcher.add_field(1, FID_DATA);
    label_idx = (label_idx + 1) % batch_label->parallel_tensor->pipe_num_part_out;
    runtime->execute_index_space(ctx, launcher);
  }
  next_label_index += ubSize;
}

void DataLoader::reset()
{
  next_index = 0;
  next_label_index = 0;
  next_input_index = 0;
  input_idx = 0;
  label_idx = 0;
}

void FlexFlow::register_custom_tasks()
{
  // Load entire dataset
  {
    TaskVariantRegistrar registrar(FlexFlow::CUSTOM_CPU_TASK_ID_1, "Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_entire_dataset>(
        registrar, "Load Entire Dataset Task");
  }
  // Load input
  {
    TaskVariantRegistrar registrar(FlexFlow::CUSTOM_GPU_TASK_ID_1, "Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Input Task");
  }
  // Load label
  {
    TaskVariantRegistrar registrar(FlexFlow::CUSTOM_GPU_TASK_ID_2, "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(
        registrar, "Load Label Task");
  }
}
