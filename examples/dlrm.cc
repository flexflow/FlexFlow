/* Copyright 2019 Stanford
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
#include <sstream>

using namespace Legion;

LegionRuntime::Logger::Category log_app("DLRM");

void parse_input_args(char **argv, int argc, DLRMConfig& apConfig);

Tensor create_mlp(FFModel* model, const Tensor& input,
                  std::vector<int> ln, int sigmoid_layer)
{
  Tensor t = input;
  for (int i = 0; i < (int)(ln.size()-1); i++) {
    float std_dev = sqrt(2.0f / (ln[i+1] + ln[i]));
    Initializer* weight_init = new NormInitializer(std::rand(), 0, std_dev);
    std_dev = sqrt(2.0f / ln[i+1]);
    Initializer* bias_init = new NormInitializer(std::rand(), 0, std_dev);
    ActiMode activation = i == sigmoid_layer ? AC_MODE_SIGMOID : AC_MODE_RELU;
    t = model->dense("linear", t, ln[i+1], activation, true/*bias*/, weight_init, bias_init);
  }
  return t;
}

Tensor create_emb(FFModel* model, const Tensor& input,
                  int input_dim, int output_dim)
{
  float range = sqrt(1.0f / input_dim);
  Initializer* embed_init = new UniformInitializer(std::rand(), -range, range);
  return model->embedding("embedding", input, input_dim, output_dim, embed_init);
}

Tensor interact_features(FFModel* model, const Tensor& x,
                         const std::vector<Tensor>& ly,
                         std::string interaction)
{
  // Currently only support cat
  // TODO: implement dot attention
  if (interaction == "cat") {
    Tensor* inputs = (Tensor*) malloc(sizeof(Tensor) * (1 + ly.size()));
    inputs[0] = x;
    for (size_t i = 0; i < ly.size(); i++)
      inputs[i+1] = ly[i];
    return model->concat("concat", ly.size() + 1, inputs, 1/*axis*/);
    free(inputs);
  } else {
    assert(false);
  }
}

void print_vector(const std::string& name, const std::vector<int>& vector)
{
  std::ostringstream out;
  for (size_t i = 0; i < vector.size() - 1; i++)
    out << vector[i] << " ";
  if (vector.size() > 0)
    out << vector[vector.size() - 1];
  log_app.print("%s: %s", name.c_str(), out.str().c_str());
}

void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
  FFConfig ffConfig;
  // Parse input arguments
  DLRMConfig dlrmConfig;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    ffConfig.parse_args(argv, argc);
    parse_input_args(argv, argc, dlrmConfig);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
        ffConfig.batchSize, ffConfig.workersPerNode, ffConfig.numNodes);
    print_vector("Embedding Size", dlrmConfig.embedding_size);
    print_vector("MLP Top", dlrmConfig.mlp_top);
    print_vector("MLP Bot", dlrmConfig.mlp_bot);
  }

  ffConfig.lg_ctx = ctx;
  ffConfig.lg_hlr = runtime;
  ffConfig.field_space = runtime->create_field_space(ctx);
  FFModel ff(ffConfig);

  std::vector<Tensor> sparse_inputs;
  for (size_t i = 0; i < dlrmConfig.embedding_size.size(); i++) {
    const int dims[] = {ffConfig.batchSize, 1};
    Tensor input = ff.create_tensor<2>(dims, "", DT_INT32);
    sparse_inputs.push_back(input);
  }
  Tensor dense_input;
  {
    const int dims[] = {ffConfig.batchSize, dlrmConfig.mlp_bot[0]};
    dense_input = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  Tensor label;
  {
    const int dims[] = {ffConfig.batchSize, 2};
    label = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  // Step 1 create dense_mlp
  Tensor x = create_mlp(&ff, dense_input, dlrmConfig.mlp_bot, dlrmConfig.sigmoid_bot);
  std::vector<Tensor> ly;
  for (size_t i = 0; i < dlrmConfig.embedding_size.size(); i++) {
    int input_dim = dlrmConfig.embedding_size[i];
    int output_dim = dlrmConfig.sparse_feature_size;
    ly.push_back(create_emb(&ff, sparse_inputs[i], input_dim, output_dim));
  }
  Tensor z = interact_features(&ff, x, ly, dlrmConfig.arch_interaction_op);
  Tensor p = create_mlp(&ff, z, dlrmConfig.mlp_top, dlrmConfig.mlp_top.size() - 2);
  if (dlrmConfig.loss_threshold > 0.0f && dlrmConfig.loss_threshold < 1.0f) {
    // TODO: implement clamp
    assert(false);
  }
  ff.mse_loss("mse_loss"/*name*/, p, label, "average"/*reduction*/);
  // Use SGD Optimizer
  ff.optimizer = new SGDOptimizer(&ff, 0.01f);
  ff.init_layers();
  // Data Loader
  DataLoader data_loader(ff, dlrmConfig, sparse_inputs, dense_input, label);
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    for (int iter = 0; iter < ffConfig.iterations; iter++) {
      data_loader.load_next_batch(ff);
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
    }
  }
}

void parse_input_args(char **argv, int argc, DLRMConfig& config)
{
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
    }
  }
}

DataLoader::DataLoader(FFModel& ff,
                       const DLRMConfig& dlrm,
                       const std::vector<Tensor>& _sparse_inputs,
                       Tensor _dense_input, Tensor _label)
{
  int num_samples;
  if (dlrm.dataset_path == "") {
    log_app.print("Use random dataset...");
    num_samples = 10;
  } else {
    assert(false);
    log_app.print("Start loading dataset from %s", dlrm.dataset_path.c_str());
    log_app.print("Finish loading dataset from %s", dlrm.dataset_path.c_str());
  }
  for (size_t i = 0; i < _sparse_inputs.size(); i++) {
    const int dims[] = {num_samples, 1};
    Tensor input = ff.create_tensor<2>(dims, "", DT_INT32);
    full_sparse_inputs.push_back(input);
    batch_sparse_inputs.push_back(_sparse_inputs[i]);
  }
  {
    batch_dense_input = _dense_input;
    const int dims[] = {num_samples, dlrm.mlp_bot[0]};
    full_dense_input = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  {
    batch_label = _label;
    const int dims[] = {num_samples, 2};
    full_label = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
}

void DataLoader::load_next_batch(FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(""));
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  for (PointInRectIterator<2> it(rect); it(); it++) {
    SampleIdxs meta;
    assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
    meta.numSamples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
    for (int i = 0; i < meta.numSamples; i++)
      meta.idxs[i] = i;
    argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
  }
  // Load Sparse Inputs
  for (size_t i = 0; i < full_sparse_inputs.size(); i++) {
    IndexLauncher launcher(CUSTOM_TASK_ID_1, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string("")));
    launcher.add_region_requirement(
        RegionRequirement(full_sparse_inputs[i].part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_sparse_inputs[i].region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_sparse_inputs[i].part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_sparse_inputs[i].region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load Dense Input
  {
    IndexLauncher launcher(CUSTOM_TASK_ID_2, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string("")));
    launcher.add_region_requirement(
        RegionRequirement(full_dense_input.part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_dense_input.region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_dense_input.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_dense_input.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load Labels
  {
    IndexLauncher launcher(CUSTOM_TASK_ID_3, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string("")));
    launcher.add_region_requirement(
        RegionRequirement(full_label.part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_label.region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_label.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_label.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
}

void DataLoader::shuffle()
{}

void register_custom_tasks()
{
  // Load Sparse Inputs
  {
    TaskVariantRegistrar registrar(CUSTOM_TASK_ID_1, "Load Sparse Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_sparse_input>(
        registrar, "Load Sparse Inputs Task");
  }
  // Load Dense Inputs
  {
    TaskVariantRegistrar registrar(CUSTOM_TASK_ID_2, "Load Densee Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_dense_input>(
        registrar, "Load Dense Inputs Task");
  }
  // Load Labels
  {
    TaskVariantRegistrar registrar(CUSTOM_TASK_ID_3, "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(
        registrar, "Load Labels");
  }
}
