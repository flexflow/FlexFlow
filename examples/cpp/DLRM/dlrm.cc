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

#include "dlrm.h"
#include "hdf5.h"
#include <sstream>

using namespace Legion;

LegionRuntime::Logger::Category log_app("DLRM");

void parse_input_args(char **argv, int argc, DLRMConfig &apConfig);

DLRMConfig::DLRMConfig(void)
    : sparse_feature_size(64), sigmoid_bot(-1), sigmoid_top(-1),
      embedding_bag_size(1), loss_threshold(0.0f), arch_interaction_op("cat"),
      dataset_path(""), data_size(-1) {
  embedding_size.push_back(1000000);
  embedding_size.push_back(1000000);
  embedding_size.push_back(1000000);
  embedding_size.push_back(1000000);
  // embedding_size.push_back(4);
  // embedding_size.push_back(4);
  mlp_bot.push_back(4);
  mlp_bot.push_back(64);
  mlp_bot.push_back(64);
  mlp_top.push_back(64);
  mlp_top.push_back(64);
  mlp_top.push_back(2);
}

Tensor create_mlp(FFModel *model,
                  Tensor const &input,
                  std::vector<int> ln,
                  int sigmoid_layer) {
  Tensor t = input;
  for (int i = 0; i < (int)(ln.size() - 1); i++) {
    float std_dev = sqrt(2.0f / (ln[i + 1] + ln[i]));
    Initializer *weight_init = new NormInitializer(std::rand(), 0, std_dev);
    std_dev = sqrt(2.0f / ln[i + 1]);
    Initializer *bias_init = new NormInitializer(std::rand(), 0, std_dev);
    ActiMode activation = i == sigmoid_layer ? AC_MODE_SIGMOID : AC_MODE_RELU;
    t = model->dense(t,
                     ln[i + 1],
                     activation,
                     false /*bias*/,
                     DT_FLOAT,
                     NULL /*weight_sharing*/,
                     weight_init,
                     bias_init);
  }
  return t;
}

Tensor create_emb(FFModel *model,
                  Tensor const &input,
                  int input_dim,
                  int output_dim,
                  int idx) {
  float range = sqrt(1.0f / input_dim);
  Initializer *embed_init = new UniformInitializer(std::rand(), -range, range);
  Tensor t = model->embedding(input,
                              input_dim,
                              output_dim,
                              AGGR_MODE_SUM,
                              DT_HALF /*dtype*/,
                              NULL /*weight_sharing*/,
                              embed_init);
  return model->cast(t, DT_FLOAT);
}

Tensor interact_features(FFModel *model,
                         Tensor const &x,
                         std::vector<Tensor> const &ly,
                         std::string interaction) {
  // Currently only support cat
  // TODO: implement dot attention
  if (interaction == "cat") {
    Tensor *inputs = (Tensor *)malloc(sizeof(Tensor) * (1 + ly.size()));
    inputs[0] = x;
    for (size_t i = 0; i < ly.size(); i++) {
      inputs[i + 1] = ly[i];
    }
    return model->concat(ly.size() + 1, inputs, -1 /*axis*/);
    free(inputs);
  } else {
    assert(false);
  }
}

void print_vector(std::string const &name, std::vector<int> const &vector) {
  std::ostringstream out;
  for (size_t i = 0; i < vector.size() - 1; i++) {
    out << vector[i] << " ";
  }
  if (vector.size() > 0) {
    out << vector[vector.size() - 1];
  }
  log_app.print("%s: %s", name.c_str(), out.str().c_str());
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffConfig;
  // Parse input arguments
  DLRMConfig dlrmConfig;
  {
    InputArgs const &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, dlrmConfig);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
                  ffConfig.batchSize,
                  ffConfig.workersPerNode,
                  ffConfig.numNodes);
    log_app.print("EmbeddingBagSize(%d)", dlrmConfig.embedding_bag_size);
    print_vector("Embedding Vocab Sizes", dlrmConfig.embedding_size);
    print_vector("MLP Top", dlrmConfig.mlp_top);
    print_vector("MLP Bot", dlrmConfig.mlp_bot);
  }

  FFModel ff(ffConfig);

  std::vector<Tensor> sparse_inputs;
  for (size_t i = 0; i < dlrmConfig.embedding_size.size(); i++) {
    int const dims[] = {ffConfig.batchSize, dlrmConfig.embedding_bag_size};
    Tensor input = ff.create_tensor<2>(dims, DT_INT64);
    sparse_inputs.push_back(input);
  }
  Tensor dense_input;
  {
    int const dims[] = {ffConfig.batchSize, dlrmConfig.mlp_bot[0]};
    dense_input = ff.create_tensor<2>(dims, DT_FLOAT);
  }
  // Tensor label;
  //{
  //   const int dims[] = {ffConfig.batchSize, 1};
  //   label = ff.create_tensor<2>(dims, DT_FLOAT);
  // }
  //  Step 1 create dense_mlp
  Tensor x =
      create_mlp(&ff, dense_input, dlrmConfig.mlp_bot, dlrmConfig.sigmoid_bot);
  std::vector<Tensor> ly;
  for (size_t i = 0; i < dlrmConfig.embedding_size.size(); i++) {
    int input_dim = dlrmConfig.embedding_size[i];
    int output_dim = dlrmConfig.sparse_feature_size;
    ly.push_back(create_emb(&ff, sparse_inputs[i], input_dim, output_dim, i));
  }
  Tensor z = interact_features(&ff, x, ly, dlrmConfig.arch_interaction_op);
  Tensor p =
      create_mlp(&ff, z, dlrmConfig.mlp_top, dlrmConfig.mlp_top.size() - 2);
  if (dlrmConfig.loss_threshold > 0.0f && dlrmConfig.loss_threshold < 1.0f) {
    // TODO: implement clamp
    assert(false);
  }
  // Use SGD Optimizer
  Optimizer *optimizer = new SGDOptimizer(&ff, 0.01f);
  std::vector<MetricsType> metrics;
  // metrics.push_back(METRICS_ACCURACY);
  // metrics.push_back(METRICS_MEAN_SQUARED_ERROR);
  ff.compile(optimizer, LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE, metrics);
  // Data Loader
  DataLoader data_loader(
      ff, dlrmConfig, sparse_inputs, dense_input, ff.label_tensor);
  ff.init_operators();

  // Warmup iterations
  for (int iter = 0; iter < 1; iter++) {
    data_loader.reset();
    ff.reset_metrics();
    data_loader.next_batch(ff);
    ff.forward();
    ff.zero_gradients();
    ff.backward();
    // ff.update();
  }

  // Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  log_app.print("Warmup finished...Start timer...");
  log_app.print("Num. epochs = %d", ffConfig.epochs);
  log_app.print("Num. iterations/epoch = %d",
                data_loader.num_samples / ffConfig.batchSize);
  printf("parameters.size() = %lu\n", ff.parameters.size());
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    data_loader.reset();
    ff.reset_metrics();
    int iterations = data_loader.num_samples / ffConfig.batchSize;
    for (int iter = 0; iter < iterations; iter++) {
      if (dlrmConfig.dataset_path.length() == 0) {
        // Only load data once for random input
        // if (iter == 0 && epoch == 0)
        //  data_loader.next_batch(ff);
      } else {
        data_loader.next_batch(ff);
      }
      // if (epoch > 0) {
      //   runtime->begin_trace(ctx, 111 /*trace_id*/);
      // }
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      // ff.update();
      // if (epoch > 0) {
      //   runtime->end_trace(ctx, 111 /*trace_id*/);
      // }
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

void parse_input_args(char **argv, int argc, DLRMConfig &config) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--arch-sparse-feature-size")) {
      config.sparse_feature_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-embedding-size")) {
      std::stringstream ss(std::string(argv[++i]));
      std::string word;
      config.embedding_size.clear();
      while (std::getline(ss, word, '-')) {
        config.embedding_size.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--embedding-bag-size")) {
      config.embedding_bag_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-mlp-bot")) {
      std::stringstream ss(std::string(argv[++i]));
      std::string word;
      config.mlp_bot.clear();
      while (std::getline(ss, word, '-')) {
        config.mlp_bot.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--arch-mlp-top")) {
      std::stringstream ss(std::string(argv[++i]));
      std::string word;
      config.mlp_top.clear();
      while (std::getline(ss, word, '-')) {
        config.mlp_top.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--loss-threshold")) {
      config.loss_threshold = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--sigmoid-top")) {
      config.sigmoid_top = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--sigmoid-bot")) {
      config.sigmoid_bot = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-interaction-op")) {
      config.arch_interaction_op = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--dataset")) {
      config.dataset_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--data-size")) {
      config.data_size = atoi(argv[++i]);
      continue;
    }
  }
}

DataLoader::DataLoader(FFModel &ff,
                       DLRMConfig const &dlrm,
                       std::vector<Tensor> const &_sparse_inputs,
                       Tensor _dense_input,
                       Tensor _label) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  num_samples = 0;
  if (dlrm.dataset_path == "") {
    log_app.print("Use random dataset...");
    if (dlrm.data_size > 0) {
      num_samples = dlrm.data_size; // num_samples = 256 * 2 * 8 * 16;
    } else {
      num_samples = 256 * 4 * ff.config.workersPerNode * ff.config.numNodes;
    }
    // num_samples = 256 * 2 * 8 * 16;
    log_app.print("Number of random samples = %d\n", num_samples);
  } else {
    log_app.print("Start loading dataset from %s", dlrm.dataset_path.c_str());
    hid_t file_id =
        H5Fopen(dlrm.dataset_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    // X_int
    {
      hsize_t dims[2], maxdims[2];
      hid_t x_int_dataset_id = H5Dopen2(file_id, "X_int", H5P_DEFAULT);
      hid_t x_int_space_id = H5Dget_space(x_int_dataset_id);
      hid_t x_int_type_id = H5Dget_type(x_int_dataset_id);
      assert(H5Sget_simple_extent_dims(x_int_space_id, dims, maxdims) == 2);
      assert(H5Tget_class(x_int_type_id) == H5T_FLOAT);
      num_samples = dims[0];
      assert(dlrm.mlp_bot[0] == (int)dims[1]);
      H5Tclose(x_int_type_id);
      H5Dclose(x_int_dataset_id);
      H5Sclose(x_int_space_id);
    }
    // X_cat
    {
      hsize_t dims[2], maxdims[2];
      hid_t x_cat_dataset_id = H5Dopen2(file_id, "X_cat", H5P_DEFAULT);
      hid_t x_cat_space_id = H5Dget_space(x_cat_dataset_id);
      hid_t x_cat_type_id = H5Dget_type(x_cat_dataset_id);
      assert(H5Sget_simple_extent_dims(x_cat_space_id, dims, maxdims) == 2);
      assert(H5Tget_class(x_cat_type_id) == H5T_INTEGER);
      assert(num_samples == (int)dims[0]);
      assert(_sparse_inputs.size() == dims[1]);
      H5Tclose(x_cat_type_id);
      H5Dclose(x_cat_dataset_id);
      H5Sclose(x_cat_space_id);
    }
    // y
    {
      hsize_t dims[2], maxdims[2];
      hid_t y_dataset_id = H5Dopen2(file_id, "y", H5P_DEFAULT);
      hid_t y_space_id = H5Dget_space(y_dataset_id);
      hid_t y_type_id = H5Dget_type(y_dataset_id);
      H5Sget_simple_extent_dims(y_space_id, dims, maxdims);
      assert(num_samples == (int)dims[0]);
      // assert(dims[1] == 1);
      H5Tclose(y_type_id);
      H5Dclose(y_dataset_id);
      H5Sclose(y_space_id);
    }
    H5Fclose(file_id);
    log_app.print("Finish loading dataset from %s", dlrm.dataset_path.c_str());
    log_app.print("Loaded %d samples", num_samples);
  }
  return;
  for (size_t i = 0; i < _sparse_inputs.size(); i++) {
    batch_sparse_inputs.push_back(_sparse_inputs[i]);
  }
  {
    int const dims[] = {num_samples,
                        (int)_sparse_inputs.size() * dlrm.embedding_bag_size};
    full_sparse_input = ff.create_tensor<2>(dims, DT_INT64);
    // ff.map_tensor(full_sparse_input, full_sparse_input->owner_op);
  }
  {
    batch_dense_input = _dense_input;
    int const dims[] = {num_samples, dlrm.mlp_bot[0]};
    full_dense_input = ff.create_tensor<2>(dims, DT_FLOAT);
    // ff.map_tensor(full_dense_input,
    // full_dense_input->parallel_tensor->owner_op);
  }
  {
    batch_label = _label;
    int const dims[] = {num_samples, 1};
    full_label = ff.create_tensor<2>(dims, DT_FLOAT);
    // ff.map_tensor(full_label, full_label->parallel_tensor->owner_op);
  }
  // Load entire dataset
  // TODO: Use index launcher instead of task launcher

  // passing DLRM Config through plain struct. ->
  ArgsConfig dlrm_args;
  assert(dlrm.embedding_size.size() <= MAX_NUM_EMB);
  assert(dlrm.dataset_path.length() <= MAX_DATASET_PATH_LEN);
  auto prev_s = dlrm.embedding_size[0];
  for (auto s : dlrm.embedding_size) {
    assert(s == prev_s);
  }
  dlrm_args.embedding_size = prev_s;
  strcpy(dlrm_args.dataset_path, dlrm.dataset_path.c_str());
  //
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1,
                        TaskArgument(&dlrm_args, sizeof(dlrm_args)));
  // regions[0]: full_sparse_input
  launcher.add_region_requirement(
      RegionRequirement(full_sparse_input->parallel_tensor->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_sparse_input->parallel_tensor->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_dense_input
  launcher.add_region_requirement(
      RegionRequirement(full_dense_input->parallel_tensor->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_dense_input->parallel_tensor->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[3]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label->parallel_tensor->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_label->parallel_tensor->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(2, FID_DATA);
  runtime->execute_task(ctx, launcher);
}

void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  // Note that these instances are in ZCM, can only use
  // TensorAccessorW with readOutput flag
  AccessorWO<int64_t, 2> const acc_sparse_input(regions[0], FID_DATA);
  AccessorWO<float, 2> const acc_dense_input(regions[1], FID_DATA);
  AccessorWO<float, 2> const acc_label_input(regions[2], FID_DATA);
  Rect<2> rect_sparse_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_dense_input = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_label_input = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(acc_sparse_input.accessor.is_dense_arbitrary(rect_sparse_input));
  assert(acc_dense_input.accessor.is_dense_arbitrary(rect_dense_input));
  assert(acc_label_input.accessor.is_dense_arbitrary(rect_label_input));
  int64_t *sparse_input_ptr = acc_sparse_input.ptr(rect_sparse_input.lo);
  float *dense_input_ptr = acc_dense_input.ptr(rect_dense_input.lo);
  float *label_input_ptr = acc_label_input.ptr(rect_label_input.lo);
  int num_samples = rect_sparse_input.hi[1] - rect_sparse_input.lo[1] + 1;
  int num_sparse_inputs = rect_sparse_input.hi[0] - rect_sparse_input.lo[0] + 1;
  assert(num_samples == rect_dense_input.hi[1] - rect_dense_input.lo[1] + 1);
  int num_dense_dims = rect_dense_input.hi[0] - rect_dense_input.lo[0] + 1;
  assert(num_samples == rect_label_input.hi[1] - rect_label_input.lo[1] + 1);
  assert(rect_label_input.hi[0] == rect_label_input.lo[0]);
  const ArgsConfig dlrm = *((ArgsConfig const *)task->args);
  int const emb_size = dlrm.embedding_size;
  std::string file_name((char const *)dlrm.dataset_path);
  if (file_name.length() == 0) {
    log_app.print("Start generating random input samples");
    for (size_t i = 0; i < rect_sparse_input.volume(); i++) {
      sparse_input_ptr[i] = std::rand() % emb_size;
    }
    for (size_t i = 0; i < rect_dense_input.volume(); i++) {
      dense_input_ptr[i] = ((float)std::rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < rect_label_input.volume(); i++) {
      label_input_ptr[i] = std::rand() % 2;
    }
  } else {
    hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    // Load X_cat
    {
      log_app.print("Start loading sparse features from "
                    "%s.%s",
                    file_name.c_str(),
                    "X_cat");
      hsize_t dims[2], maxdims[2];
      hid_t x_cat_dataset_id = H5Dopen2(file_id, "X_cat", H5P_DEFAULT);
      hid_t x_cat_space_id = H5Dget_space(x_cat_dataset_id);
      hid_t x_cat_type_id = H5Dget_type(x_cat_dataset_id);
      assert(H5Sget_simple_extent_dims(x_cat_space_id, dims, maxdims) == 2);
      assert(H5Tget_class(x_cat_type_id) == H5T_INTEGER);
      assert(num_samples == (int)dims[0]);
      assert(num_sparse_inputs == (int)dims[1]);
      H5Dread(x_cat_dataset_id,
              H5T_NATIVE_LLONG,
              H5S_ALL,
              H5S_ALL,
              H5P_DEFAULT,
              sparse_input_ptr);
      H5Tclose(x_cat_type_id);
      H5Dclose(x_cat_dataset_id);
      H5Sclose(x_cat_space_id);
      log_app.print("Finish loading sparse features");
    }
    // Load X_int
    {
      log_app.print("Start loading dense features from "
                    "%s.%s",
                    file_name.c_str(),
                    "X_int");
      hsize_t dims[2], maxdims[2];
      hid_t x_int_dataset_id = H5Dopen2(file_id, "X_int", H5P_DEFAULT);
      hid_t x_int_space_id = H5Dget_space(x_int_dataset_id);
      hid_t x_int_type_id = H5Dget_type(x_int_dataset_id);
      assert(H5Sget_simple_extent_dims(x_int_space_id, dims, maxdims) == 2);
      assert(H5Tget_class(x_int_type_id) == H5T_FLOAT);
      num_samples = dims[0];
      assert(num_dense_dims == (int)dims[1]);
      H5Dread(x_int_dataset_id,
              H5T_NATIVE_FLOAT,
              H5S_ALL,
              H5S_ALL,
              H5P_DEFAULT,
              dense_input_ptr);
      H5Tclose(x_int_type_id);
      H5Dclose(x_int_dataset_id);
      H5Sclose(x_int_space_id);
      log_app.print("Finish loading dense features");
    }
    // Load y
    {
      log_app.print("Start loading labels from "
                    "%s.%s",
                    file_name.c_str(),
                    "y");
      hsize_t dims[2], maxdims[2];
      hid_t y_dataset_id = H5Dopen2(file_id, "y", H5P_DEFAULT);
      hid_t y_space_id = H5Dget_space(y_dataset_id);
      hid_t y_type_id = H5Dget_type(y_dataset_id);
      H5Sget_simple_extent_dims(y_space_id, dims, maxdims);
      assert(num_samples == (int)dims[0]);
      // assert(dims[1] == 1);
      H5Dread(y_dataset_id,
              H5T_NATIVE_FLOAT,
              H5S_ALL,
              H5S_ALL,
              H5P_DEFAULT,
              label_input_ptr);
      H5Tclose(y_type_id);
      H5Dclose(y_dataset_id);
      H5Sclose(y_space_id);
      log_app.print("Finish loading labels");
    }
  }
}

void DataLoader::next_batch(FFModel &ff) {
  return;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Load Sparse Inputs
  for (size_t i = 0; i < batch_sparse_inputs.size(); i++) {
    int hash = batch_sparse_inputs.size() * MAX_NUM_EMB + i;
    Domain domain = runtime->get_index_space_domain(
        ctx, batch_sparse_inputs[i]->parallel_tensor->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize ==
             batch_sparse_inputs[i]->parallel_tensor->dims[1].size);
      meta.num_samples =
          ff.config.batchSize /
          batch_sparse_inputs[i]->parallel_tensor->dims[1].degree;
      // Assert that we have enough slots to save the indices
      assert(meta.num_samples <= MAX_NUM_SAMPLES);
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(
        CUSTOM_GPU_TASK_ID_1,
        batch_sparse_inputs[i]->parallel_tensor->parallel_is,
        TaskArgument(&hash, sizeof(int)),
        argmap,
        Predicate::TRUE_PRED,
        false /*must*/,
        0 /*mapper_id*/,
        batch_sparse_inputs[i]->parallel_tensor->machine_view.hash());
    // Full dataset in ZCM
    launcher.add_region_requirement(
        RegionRequirement(full_sparse_input->parallel_tensor->region,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          full_sparse_input->parallel_tensor->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_sparse_inputs[i]->parallel_tensor->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_sparse_inputs[i]->parallel_tensor->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load Dense Input
  {
    Domain domain = runtime->get_index_space_domain(
        ctx, batch_dense_input->parallel_tensor->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize ==
             batch_dense_input->parallel_tensor->dims[1].size);
      meta.num_samples = ff.config.batchSize /
                         batch_dense_input->parallel_tensor->dims[1].degree;
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(
        CUSTOM_GPU_TASK_ID_2,
        batch_dense_input->parallel_tensor->parallel_is,
        TaskArgument(NULL, 0),
        argmap,
        Predicate::TRUE_PRED,
        false /*must*/,
        0 /*mapper_id*/,
        batch_dense_input->parallel_tensor->machine_view.hash());
    // Full dataset in ZCM
    launcher.add_region_requirement(
        RegionRequirement(full_dense_input->parallel_tensor->region,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          full_dense_input->parallel_tensor->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_dense_input->parallel_tensor->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_dense_input->parallel_tensor->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load Labels
  {
    Domain domain = runtime->get_index_space_domain(
        ctx, batch_label->parallel_tensor->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % batch_label->parallel_tensor->dims[1].size);
      meta.num_samples =
          ff.config.batchSize / batch_label->parallel_tensor->dims[1].degree;
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_3,
                           batch_label->parallel_tensor->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_label->parallel_tensor->machine_view.hash());
    // Full dataset in ZCM
    launcher.add_region_requirement(
        RegionRequirement(full_label->parallel_tensor->region,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          full_label->parallel_tensor->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_label->parallel_tensor->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_label->parallel_tensor->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // progress next_index
  next_index += ff.config.batchSize;
}

void DataLoader::shuffle() {}

void DataLoader::reset() {
  next_index = 0;
}

void DataLoader::load_sparse_input_cpu(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  std::cout << "load_sparse_input_cpu" << std::endl;
}

void FlexFlow::register_custom_tasks() {
  // Load entire dataset
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_1, "Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_entire_dataset>(
        registrar, "Load Entire Dataset Task");
  }
  // Load Sparse Inputs
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "Load Sparse Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_sparse_input>(
        registrar, "Load Sparse Inputs Task");
  }
  // Load Dense Inputs
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2, "Load Dense Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_dense_input>(
        registrar, "Load Dense Inputs Task");
  }
  // Load Labels
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_3, "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(registrar,
                                                              "Load Labels");
  }
}
