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
#include "moe.h"
#include <fstream>
#include <string>
#include <cmath>
#include <sstream>

#define SET_ALPHA(a,b,c,d,e,f,g,h) new_alpha_vec.push_back(a);new_alpha_vec.push_back(b);new_alpha_vec.push_back(c);new_alpha_vec.push_back(d);new_alpha_vec.push_back(e);new_alpha_vec.push_back(f);new_alpha_vec.push_back(g);new_alpha_vec.push_back(h);

using namespace Legion;


LegionRuntime::Logger::Category log_app("MoE");
int num_exp = 3;
int num_select = 1;
int epoch = 0;
int recompiles = 0; // TODO: Comment out, use the one of recompile state

int glob_trace_id = 111;

void parse_input_args(char **argv, int argc, MoeConfig& config)
{
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--dataset")) {
      config.dataset_path = std::string(argv[++i]);
      continue;
    }
  }
}


// =============================================================================
//  User-defined functions on using cached expert assignments
// =============================================================================

// Score: Running average over sample ratio of which experts are corr. cached
float moe_score(float* cached_score,
                const void* input,
                const void* cached,
                int vol) {
  float gamma = 0.99f;
  *cached_score *= gamma;
  int* cast_input = (int*)input;
  int* cast_cached = (int*)cached;
  int batch_size = vol/num_select;
  float frac = (1.0f-gamma)/batch_size;
  for(int i = 0; i < batch_size; i++) {
    std::set<int, std::greater<int>> cached;
    std::set<int, std::greater<int>> input;
    for(int j = 0; j < num_select; j++) {
      cached.insert(cast_input[i*num_select+j]);
      input.insert(cast_cached[i*num_select+j]);
    }
    if(cached == input)
      *cached_score += frac;
  }
  printf("score: %.3f\n", *cached_score);
  return *cached_score;
}

float moe_score_gar(float* cached_score,
                const void* input,
                const void* cached,
                int vol) {
  return 0.0f;
}



bool moe_alter(FFModel* ff, RecompileState& r) {
  // cache recompile hyperparams
  float cache_thresh_up = 0.95f;
  float cache_thresh_low = 0.85f;

  // capacity factor recompile hyperparamss
  float groupby_thresh_max = 1.0f;
  float groupby_overhead_max = 1.3f;
  float max_factor = 4.0f;

  // get layers and scores
  std::vector<int> topk_idx;
  std::vector<int> cache_idx;
  std::vector<int> groupby_idx;
  std::vector<int> agg_idx;
  std::vector<int> aggspec_idx;

  std::vector<float> cache_score;
  std::vector<float*> groupby_score;
  // get cache score and groupby max
  // FIXME: if cache or group by split onto several devices
  for(size_t i = 0; i < ff->layers.size(); i++) {
    switch(ff->layers[i]->op_type) {
      case OP_TOPK:
        topk_idx.push_back(i);
        break;
      case OP_CACHE:
        cache_idx.push_back(i);
        cache_score.push_back(((Cache*)ff->layers[i])->score_futures.front()
          .get_result<float>());
        ((Cache*)ff->layers[i])->score_futures.pop_front();
        break;
      case OP_GROUP_BY:
        groupby_idx.push_back(i);
        // groupby_score.push_back(((GroupBy*)ff->layers[i])->score_futures.front()
        //   .get_result<float*>());
        // ((GroupBy*)ff->layers[i])->score_futures.pop_front();
        break;
      case OP_AGGREGATE:
        agg_idx.push_back(i);
        break;
      case OP_AGG_SPEC:
        aggspec_idx.push_back(i);
        break;
      default:
        break;
    }
  }
  // recompile
  bool rec = false;

  // FIXME: for now, no hierarchical models
  // assert(cache_score.size() == 1 && groupby_score.size() == 1);

  // FIXME: If no aggspec
  int topk_layer = topk_idx[0];
  int cache_layer = cache_idx[0];
  int groupby_layer = groupby_idx[0];
  int agg_layer = agg_idx[0];
  int aggspec_layer = aggspec_idx[0];

  // cache recompile

  // trigger based on first recompile. alter last num_exp recompiles
  if(!((Cache*)ff->layers[cache_layer])->load_cached &&
    cache_score[0] > cache_thresh_up && r.last_recompile > 2000) {
    printf("alter cache!!\n");
    rec = true;

    // first cache output is used for aggregate
    // int cache_layer = cache_idx[0];
    ((Cache*)ff->layers[cache_layer])->use_cached(true);
    // Aggregate input
    ff->layers[agg_layer]->inputs[1] = ff->layers[cache_layer]->outputs[0];
    ff->layers[agg_layer]->input_lps[1] = ff->layers[cache_layer]->outputs[0].part;
    ff->layers[agg_layer]->input_grad_lps[1] = ff->layers[cache_layer]->outputs[0].part_grad;
    // AggregateSpec input
    ff->layers[aggspec_layer]->inputs[1] = ff->layers[cache_layer]->outputs[0];
    ff->layers[aggspec_layer]->input_lps[1] = ff->layers[cache_layer]->outputs[0].part;
    ff->layers[aggspec_layer]->input_grad_lps[1] = ff->layers[cache_layer]->outputs[0].part_grad;

    // last num_exp cache outputs are used for expert layers
    for(int i = 1; i < cache_idx.size(); i++) {
      cache_layer = cache_idx[i];
      ((Cache*)ff->layers[cache_layer])->use_cached(true);

      // Group by input
      ff->layers[cache_layer+1]->inputs[0] = ff->layers[cache_layer]->outputs[0];
      ff->layers[cache_layer+1]->input_lps[0] = ff->layers[cache_layer]->outputs[0].part;
      ff->layers[cache_layer+1]->input_grad_lps[0] = ff->layers[cache_layer]->outputs[0].part_grad;
    }
  }
  else if(((Cache*)ff->layers[cache_layer])->load_cached &&
    cache_score[0] < cache_thresh_low) {
    printf("alter cache!!\n");
    ((Cache*)ff->layers[cache_layer])->use_cached(false);

    // last num_exp cache outputs are used for expert layers
    for(int i = 1; i < cache_idx.size(); i++) {
      cache_layer = cache_idx[i];
      ((Cache*)ff->layers[cache_layer])->use_cached(false);

      // Group by input
      ff->layers[cache_layer+1]->inputs[0] = ff->layers[groupby_layer]->outputs[i-1];
      ff->layers[cache_layer+1]->input_lps[0] = ff->layers[groupby_layer]->outputs[i-1].part;
      ff->layers[cache_layer+1]->input_grad_lps[0] = ff->layers[groupby_layer]->outputs[i-1].part_grad;
    }

    rec = true;

    // Aggregate input
    ff->layers[agg_layer]->inputs[1] = ff->layers[topk_layer]->outputs[1];
    ff->layers[agg_layer]->input_lps[1] = ff->layers[topk_layer]->outputs[1].part;
    ff->layers[agg_layer]->input_grad_lps[1] = ff->layers[topk_layer]->outputs[1].part_grad;
    // AggregateSpec input
    ff->layers[aggspec_layer]->inputs[1] = ff->layers[topk_layer]->outputs[1];
    ff->layers[aggspec_layer]->input_lps[1] = ff->layers[topk_layer]->outputs[1].part;
    ff->layers[aggspec_layer]->input_grad_lps[1] = ff->layers[topk_layer]->outputs[1].part_grad;
  }

  // capacity factor recompile

  // increment trace id if triggered
  // TODO: What if test set ....
  if(rec) {
    glob_trace_id++;
  }
  return rec;
}


void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{

  printf("heeeeree\n");

  FFConfig ffConfig;
  MoeConfig moeConfig;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, moeConfig);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
        ffConfig.batchSize, ffConfig.workersPerNode, ffConfig.numNodes);
  }
  FFModel ff(ffConfig);

  Tensor input;
  {
    const int dims[] = {ffConfig.batchSize, INPUT_DIM};
    input = ff.create_tensor<D_DIM>(dims, DT_FLOAT);
  }



//-----------------------------------------------------------------

  GlorotUniform* kernel_initializer = new GlorotUniform(4);
  ZeroInitializer* bias_initializer = new ZeroInitializer();

  float alpha = 2.0f;
  // float lambda = 0.01; // coop loss cifar10
  float lambda = 0.06f/60.0f; //0.04/60.0f; // spec loss cifar100

  // MoE model
// #ifdef USE_CNN
  // Tensor t = ff.conv2d(input, 64, 11, 11, 4, 4, 2, 2, AC_MODE_RELU);
  // t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  // t = ff.conv2d(t, 192, 5, 5, 1, 1, 2, 2, AC_MODE_RELU);
  // t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  // t = ff.flat(t);
  // Tensor gate_preds = ff.dense(t, 80, AC_MODE_SIGMOID);
// #else
  Tensor t = ff.dense(input, 256, AC_MODE_RELU, true, NULL, kernel_initializer, bias_initializer, NULL);
  t = ff.dense(t, 256, AC_MODE_RELU, true, NULL, kernel_initializer, bias_initializer, NULL);
  Tensor gate_preds = ff.dense(t, 128, AC_MODE_RELU, true, NULL, kernel_initializer, bias_initializer, NULL);
// #endif
  gate_preds = ff.dense(gate_preds, 64, AC_MODE_SIGMOID);
  gate_preds = ff.dense(gate_preds, num_exp, AC_MODE_SIGMOID);
  Tensor soft_gate_preds = ff.softmax(gate_preds);

  Tensor topK_output[2];
  ff.top_k(gate_preds, topK_output, num_select, false);
  // ff.cache(topK_output[1], (TRAIN_SAMPLES+TEST_SAMPLES) / ffConfig.batchSize, moe_score);

  Tensor exp_tensors[num_exp];
  ff.group_by(input, topK_output[1], exp_tensors, num_exp, alpha);

  Tensor agg_inputs[num_exp+4];
  agg_inputs[0] = ff.softmax(topK_output[0]); // gate preds
  agg_inputs[1] = topK_output[1]; // gate assign
  agg_inputs[2] = topK_output[1]; // gate assign TopK (for cache)
  agg_inputs[3] = soft_gate_preds; // full gate preds
  for(int i = 0; i < num_exp; i++) {
// #ifdef USE_CNN
    // ff.cache(exp_tensors[i], (TRAIN_SAMPLES+TEST_SAMPLES) / ffConfig.batchSize, moe_score_gar);
    // Tensor t = ff.conv2d(exp_tensors[i], 64, 11, 11, 4, 4, 2, 2, AC_MODE_RELU);
    // t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
    // t = ff.conv2d(t, 192, 5, 5, 1, 1, 2, 2, AC_MODE_RELU);
    // t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
    // t = ff.flat(t);
    // t = ff.dense(t, 64, AC_MODE_RELU);
// #else
    Tensor t = ff.dense(exp_tensors[i], 257, AC_MODE_RELU);
    t = ff.dense(t, 256, AC_MODE_RELU);
    t = ff.dense(t, 128, AC_MODE_RELU);
    t = ff.dense(t, 64, AC_MODE_RELU);
// // #endif
    Tensor exp_pred = ff.dense(t, OUT_DIM, AC_MODE_RELU);
    agg_inputs[i+4] = ff.softmax(exp_pred);
    // agg_inputs[i+4] = exp_pred;

  }

  Tensor coop_output = ff.aggregate(agg_inputs, num_exp, lambda);
  ff.get_metrics();
  Tensor final_pred = ff.aggregate_spec(agg_inputs, num_exp, lambda);



//-----------------------------------------------------------------

  Optimizer* optimizer = new SGDOptimizer(&ff, 0.002f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);

  // Data Loader
  DataLoader data_loader(ff, moeConfig, input, ff.label_tensor);
  RecompileState r(&ff, &moe_alter);
  printf("hello\n");
  ff.init_layers();

  // ff.load("a3/debugmorning.ff");


  // ff.load("a3/c10n5k2l5d40-norec.ff");

  assert(TRAIN_SAMPLES % ffConfig.batchSize == 0 &&
    TEST_SAMPLES % ffConfig.batchSize == 0);

  //Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();
  printf("here\n");
  for (epoch = 0; epoch < ffConfig.epochs; epoch++) {
    data_loader.reset();
    ff.reset_metrics();
    int iterations = TRAIN_SAMPLES / ffConfig.batchSize;

    for (int iter = 0; iter < iterations; iter++) {
      data_loader.next_batch(ff);
      // if (epoch > 0) {
      //   runtime->begin_trace(ctx, glob_trace_id/*trace_id*/);
      // }
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
      ff.recompile_on_condition(r);
      // if (epoch > 0) {
      //   runtime->end_trace(ctx, glob_trace_id/*trace_id*/);
      // }
    }

    // TODO: Do properly
    ff.reset_metrics();
    iterations = TEST_SAMPLES / ffConfig.batchSize;
    for (int iter = 0; iter < iterations; iter++) {
      data_loader.next_batch(ff);
      ff.forward_test();
      ff.recompile_on_condition(r, false);
    }

    // ff.store("a3/debugmorning.ff");
  }
  printf("done %d\n", ffConfig.epochs);

  // End timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double run_time = 1e-6 * (ts_end - ts_start);
  printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s (comb. train, test)\n", run_time,
         NUM_SAMPLES * ffConfig.epochs / run_time);

  // ff.store("a3/n3k1lr2l5d60-95-84-cache-dropwrong-try3.ff");
  // ff.store("a3/c10n5k2l5d60-cache-try1.ff");

}

DataLoader::DataLoader(FFModel& ff, const MoeConfig& moe,
                       Tensor input, Tensor label)
{
  num_samples = NUM_SAMPLES;

  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;

  // Create full input
  {
    batch_input = input;
    const int dims[] = {NUM_SAMPLES, INPUT_DIM};
    full_input = ff.create_tensor<D_DIM>(dims, DT_FLOAT);
  }
  // Create full label
  {
    batch_label = label;
    const int dims[] = {NUM_SAMPLES, label.adim[0]};
    full_label = ff.create_tensor<2>(dims, DT_INT32);
  }

  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  const MoeConfig* ptr = &moe;
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1,
      TaskArgument(&ptr, sizeof(MoeConfig*)));
  // regions[0]: full_input
  launcher.add_region_requirement(
      RegionRequirement(full_input.region, WRITE_ONLY,
                        EXCLUSIVE, full_input.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label.region, WRITE_ONLY,
                        EXCLUSIVE, full_label.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);

  runtime->execute_task(ctx, launcher);
  reset();
  next_batch(ff);
}

__inline__
int calc_offset(int c, int y, int x, int yscale, int xscale)
{
  return (c * yscale * xscale + y * xscale + x);
}


// =================================================
//                    Load data
// =================================================

/* NOTE: Download files from http://yann.lecun.com/exdb/mnist/, unpack to
this directory (Flexflow/examples/cpp/mixture_of_experts) */


void read_cifar100(float* input_ptr, int* label_ptr) {
  vector<std::string> files{ "train.bin", "test.bin" };
  vector<int> sample_sizes = {50000, 10000};

  int sample = 0;

  for(int f = 0; f < files.size(); f++) {
    std::ifstream file;
    file.open(files[f], std::ios::in | std::ios::binary | std::ios::ate);
    if (!file) {
        std::cout << "Error opening CIFAR100 data file " << files[f] << std::endl;
        assert(false);
    }

    file.seekg(0, std::ios::beg);

    // each sample: <1 x coarse label><1 x fine label><3072 x pixel>
    for(std::size_t i = 0; i < sample_sizes[f]; i++) {
      unsigned char temp = 0;
      file.read((char*)&temp, sizeof(temp)); // coarse label, skip
      file.read((char*)&temp, sizeof(temp));
      label_ptr[sample] = temp;
      for(std::size_t j = 0; j < 3072; ++j) {
        file.read((char*)&temp, sizeof(temp));
        input_ptr[sample*3072 + j] = (float)temp/255.0f;
      }
      sample++;
    }

    file.close();
  }
}

void read_cifar100_c(float* input_ptr, int* label_ptr) {
  std::ifstream file;
  file.open("train.bin", std::ios::in | std::ios::binary | std::ios::ate);
  if (!file) {
      std::cout << "Error opening CIFAR100 train data file" << std::endl;
      assert(false);
  }

  file.seekg(0, std::ios::beg);

  // each sample: <1 x coarse label><1 x fine label><3072 x pixel>
  for(std::size_t i = 0; i < NUM_SAMPLES; i++) {
    unsigned char temp = 0;
    file.read((char*)&temp, sizeof(temp));
    label_ptr[i] = temp;
    file.read((char*)&temp, sizeof(temp)); // fine label, skip
    for(std::size_t j = 0; j < 3072; ++j) {
      file.read((char*)&temp, sizeof(temp));
      input_ptr[i*3072 + j] = (float)temp/255.0f;
    }
  }

  file.close();
}

void read_cifar10(float* input_ptr, int* label_ptr) {
  vector<std::string> files{ "data_batch_1.bin",
                             "data_batch_2.bin",
                             "data_batch_3.bin",
                             "data_batch_4.bin",
                             "data_batch_5.bin",
                             "test_batch.bin" };

  int sample = 0;

  for(int f = 0; f < files.size(); f++) {
    std::ifstream file;
    file.open(files[f], std::ios::in | std::ios::binary | std::ios::ate);
    if (!file) {
        std::cout << "Error opening CIFAR10 data file " << files[f] << std::endl;
        assert(false);
    }

    file.seekg(0, std::ios::beg);

    // each sample: <1 x coarse label><1 x fine label><3072 x pixel>
    for(std::size_t i = 0; i < 10000; i++) {
      unsigned char temp = 0;
      file.read((char*)&temp, sizeof(temp));
      label_ptr[sample] = temp;
      for(std::size_t j = 0; j < 3072; ++j) {
        file.read((char*)&temp, sizeof(temp));
        input_ptr[sample*3072 + j] = (float)temp/255.0f;
      }
      sample++;
    }

    file.close();
  }
}


int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


void read_mnist(float* input_ptr, int* label_ptr)
{
  // read inputs
  std::ifstream input("train-images-idx3-ubyte", std::ios::binary);
  if (input.is_open())
  {
    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;
    input.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
    input.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    input.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    input.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);

    for(int i = 0; i < number_of_images; i++) {
      for(int r = 0; r < n_rows; r++) {
        for(int c = 0; c < n_cols; c++) {
          unsigned char temp=0;
          input.read((char*)&temp,sizeof(temp));
          input_ptr[i*n_rows*n_cols + r*n_cols + c] = (float)temp/255.0f;
        }
      }
    }
  }
  else {
    std::cout << "Error opening MNIST input data file" << std::endl;
    assert(false);
  }

  // read labels
  std::ifstream labels("train-labels-idx1-ubyte", std::ios::binary);
  if (labels.is_open())
  {
    int magic_number=0;
    int number_of_images=0;
    labels.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
    labels.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);

    for(int i = 0; i < number_of_images; i++) {
      unsigned char temp = 0;
      labels.read((char*)&temp, sizeof(temp));
      label_ptr[i] = temp;
    }
  }
  else {
    std::cout << "Error opening MNIST label data file" << std::endl;
    assert(false);
  }
}


void DataLoader::load_entire_dataset(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime* runtime)
{
  //const MoeConfig* conf = *((MoeConfig**)task->args);
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());

  // get input and label pointer
  const AccessorWO<float, D_DIM> acc_input(regions[0], FID_DATA);
  const AccessorWO<int, 2> acc_label(regions[1], FID_DATA);
  Rect<D_DIM> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  Rect<2> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  float* input_ptr = acc_input.ptr(rect_input.lo);
  int* label_ptr = acc_label.ptr(rect_label.lo);

  READ_DATA(input_ptr, label_ptr);
  log_app.print("finish loading data\n");
}


void DataLoader::next_batch(FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // Load input
  {
    IndexSpaceT<D_DIM> task_is = IndexSpaceT<D_DIM>(ff.get_or_create_task_is(D_DIM, ""));
    Rect<D_DIM> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<D_DIM> it(rect); it(); it++) {
      SampleIdxs meta;
      // printf("%d %d and %d\n", D_DIM, ff.config.batchSize, (rect.hi[D_DIM-1] - rect.lo[D_DIM-1] + 1));
      assert(ff.config.batchSize % (rect.hi[D_DIM-1] - rect.lo[D_DIM-1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[D_DIM-1] - rect.lo[D_DIM-1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1, task_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(""));
    launcher.add_region_requirement(
        RegionRequirement(full_input.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_input.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_input.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_input.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load label
  {
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(2, ""));
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
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2, task_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(""));
    launcher.add_region_requirement(
        RegionRequirement(full_label.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_label.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_label.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_label.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  next_index += ff.config.batchSize;
}

void DataLoader::reset()
{
  next_index = 0;
}

void register_custom_tasks()
{
  // Load entire dataset
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_1, "Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_entire_dataset>(
        registrar, "Load Entire Dataset Task");
  }
  // Load input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Input Task");
  }
  // Load label
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2, "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(
        registrar, "Load Label Task");
  }
}
