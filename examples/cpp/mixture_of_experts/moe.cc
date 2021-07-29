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
int num_exp = 8;
int num_select = 2;
int epoch = 0;
int recompiles = 0;

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

// Trigger: If average score of all cache layers is above thresh
bool moe_trigger(FFModel* ff) {
  float thresh = 1.0f;

  int num_futures = 0;
  float score = 0.0f;
  for(size_t i = 0; i < ff->layers.size(); i++) {
    if(ff->layers[i]->op_type == OP_CACHE) {
      int num_futures_i = ((Cache*)ff->layers[i])->score_futures.size();
      num_futures += num_futures_i;
      for(int j = 0; j < num_futures_i; j++)
        score += ((Cache*)ff->layers[i])->score_futures[j].get_result<float>();
    }
  }
  return score >= thresh;
}

#ifdef MOE_CF_LOCAL
// Alter: GroupBy, Aggregate, AggregateSpec use cached values for expert assign.
void moe_alter(FFModel* ff) {
  float cache_thresh = 1.0f;
  float groupby_thresh_max = 1.0f; // 1.0f
  float groupby_overhead_max = 1.3f;
  float cache_score = 0.0f;
  float max_factor = 4.0f;
  std::vector<int> groupby_idx;
  std::vector<float*> groupby_max;

  // get cache score and groupby max
  // TODO: normalize scores (divide by amt of futures)
  for(size_t i = 0; i < ff->layers.size(); i++) {
    // if(ff->layers[i]->op_type == OP_CACHE) {
    //   std::deque<Future>* futures = &((Cache*)ff->layers[i])->score_futures;
    //   // TODO: Here only pop one per partition
    //   while(!futures->empty()) {
    //     cache_score += futures->front().get_result<float>();
    //     futures->pop_front();
    //   }
    // }
    if(ff->layers[i]->op_type == OP_GROUP_BY) {
      std::deque<Future>* futures = &((GroupBy*)ff->layers[i])->score_futures;
      groupby_idx.push_back(i);
      float* gb_alphas = futures->front().get_result<float*>();
      futures->pop_front();
      groupby_max.push_back(gb_alphas);
    }
  }

  // printf("GB: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", groupby_max[0][0], groupby_max[0][1], groupby_max[0][2], groupby_max[0][3], groupby_max[0][4], groupby_max[0][5], groupby_max[0][6], groupby_max[0][7]);

  // TODO: alter du musst auch auf scheiss 543 schauen.

  // alpha 3.0, max accuracy!
  // bool do_it;
  //
  // std:vector<float> new_alpha_vec;
  // switch(recompiles)
  // {
  //   case 0:
  //     // epochs 5-60
  //     do_it = true;
  //     for(int i = 0; i < 8; i++)
  //       if(groupby_max[0][i] > 1.58f)
  //         do_it = false;
  //     if(do_it) {
  //       SET_ALPHA(1.7f, 1.6f, 1.5f, 1.75f, 1.6f, 1.7f, 1.5f, 1.5f)
  //     }
  //     break;
  //   case 1:
  //     // 60 - 120
  //     if(groupby_max[0][1] > 1.7f || groupby_max[0][0] > 1.8f) {
  //       // SET_ALPHA(2.2f, 2.3f, 1.3f, 1.6f, 1.5f, 1.5f, 1.35f, 1.5f) // at around 60-70
  //       SET_ALPHA(2.1f, 2.0f, 1.3f, 1.72f, 1.4f, 1.55f, 1.4f, 1.25f)
  //     }
  //     break;
  //   case 2:
  //     // 120 - 200
  //     if(groupby_max[0][7] < 1.1f && groupby_max[0][2] < 1.1f && epoch > 115) { // maybe also triggers based on 6 4 2
  //       // SET_ALPHA(2.2f, 2.3f, 1.0f /*TODO*/, 1.6f, 1.3f, 1.5f, 1.5f, 1.1f) // at around 100
  //       SET_ALPHA(2.3f, 2.3f, 1.2f, 1.6f, 1.4f, 1.6f, 1.35f, 1.15f)
  //     }
  //     break;
  //   case 3:
  //     //  200-335
  //     if(epoch > 190 && ((groupby_max[0][7] < 1.0f && groupby_max[0][2] < 1.0f) || groupby_max[0][1] > 1.65f)) {
  //       SET_ALPHA(2.4f, 2.3f, 1.0f, 1.7f, 1.4f, 1.8f, 1.25f, 1.0f) // at around 200+
  //     }
  //     break;
  //   case 4:
  //     // 335-440
  //     if(groupby_max[0][0] > 2.45f) {
  //       SET_ALPHA(2.8f, 2.4f, 0.6f, 1.6f, 1.3f, 1.9f, 1.3f, 0.65f)
  //     }
  //     break;
  //   default:
  //     break;
  // }

  // alpha 1.0
  std:vector<float> new_alpha_vec;
  switch(recompiles)
  {
    case 0:
      // epochs 5-60
      if(epoch == 60) {
        SET_ALPHA(1.34, 1.3, 0.82, 1.0, 0.87, 0.92, 0.95, 0.8)
      }
      break;
    case 1:
      // 60 - 120
      if(epoch == 110) {
        SET_ALPHA(1.6f, 1.5f, 0.73f, 0.95f, 0.82f, 0.85f, 0.85f, 0.7f)
      }
      break;
    case 2:
      // 120 - 200
      if(epoch == 150) { // maybe also triggers based on 6 4 2
        SET_ALPHA(1.75f, 1.73f, 0.6f, 0.85f, 0.75f, 0.9f, 0.75f, 0.67f)
      }
      break;
    case 3:
      //  200-335
      if(epoch == 250) {
        SET_ALPHA(1.81f, 1.73f, 0.39f, 0.95f, 0.75f, 1.05f, 0.82f, 0.5f) // at around 200+
      }
      break;
    case 4:
      // 335-440
      if(epoch == 400) {
        SET_ALPHA(2.05f, 1.8f, 0.19f, 1.0f, 0.67f, 1.15f, 0.85f, 0.29f)
      }
      break;
    case 5:
      // 335-440
      if(epoch == 470) {
        SET_ALPHA(2.2f, 1.9f, 0.12f, 1.0f, 0.58f, 1.18f, 0.83f, 0.19f)
      }
      break;
    default:
      break;
  }


  if(new_alpha_vec.size() > 0) {
    float al_sum = 0.0f;
    for(int j = 0; j < 8; j++) {
      al_sum += new_alpha_vec[j];
    }
    printf("\n\nalter alphas:");
    for(int j = 0; j < 8; j++) {
      ((GroupBy*)ff->layers[groupby_idx[0]])->alpha[j] = new_alpha_vec[j];
      printf(" %.5f (%.5f)", new_alpha_vec[j], groupby_max[0][j]);
    }
    printf("\nsum: %.2f; alpha: %.2f; epoch: %d\n\n\n", al_sum, al_sum/8, epoch);
    printf("\n\n\n");
    free(groupby_max[0]);

    vector<int> changed_layers;
    changed_layers.push_back(9);
    ff->recompile(changed_layers);
    glob_trace_id++;
    recompiles++;
  }
  else {
    free(groupby_max[0]);
  }


  return;

  // // dermine if cache trigger
  // if(cache_score > cache_thresh && false) {
  //   printf("alter cache!!\n");
  //   ((Cache*)ff->layers[4])->use_cached(true);
  //   // Group by input
  //   ff->layers[5]->inputs[1] = ff->layers[4]->outputs[0];
  //   ff->layers[5]->input_lps[1] = ff->layers[4]->outputs[0].part;
  //   ff->layers[5]->input_grad_lps[1] = ff->layers[4]->outputs[0].part_grad;
  //   // Aggregate input
  //   ff->layers[22]->inputs[1] = ff->layers[4]->outputs[0];
  //   ff->layers[22]->input_lps[1] = ff->layers[4]->outputs[0].part;
  //   ff->layers[22]->input_grad_lps[1] = ff->layers[4]->outputs[0].part_grad;
  //   // AggregateSpec input
  //   ff->layers[23]->inputs[1] = ff->layers[4]->outputs[0];
  //   ff->layers[23]->input_lps[1] = ff->layers[4]->outputs[0].part;
  //   ff->layers[23]->input_grad_lps[1] = ff->layers[4]->outputs[0].part_grad;
  //
  //   remap = true;
  // }

  // return remap;
}
#else
// Alter: GroupBy, Aggregate, AggregateSpec use cached values for expert assign.
void moe_alter(FFModel* ff) {
  float cache_thresh = 1.0f;
  float groupby_thresh_max = 1.0f; // 1.0f
  float groupby_overhead_max = 1.3f;
  float cache_score = 0.0f;
  float max_factor = 4.0f;
  std::vector<int> groupby_idx;
  std::vector<float> groupby_max;

  // get cache score and groupby max
  // TODO: normalize scores (divide by amt of futures)
  for(size_t i = 0; i < ff->layers.size(); i++) {
    if(ff->layers[i]->op_type == OP_CACHE) {
      std::deque<Future>* futures = &((Cache*)ff->layers[i])->score_futures;
      // TODO: Here only pop one per partition
      while(!futures->empty()) {
        cache_score += futures->front().get_result<float>();
        futures->pop_front();
      }
    }
    else if(ff->layers[i]->op_type == OP_GROUP_BY) {
      std::deque<Future>* futures = &((GroupBy*)ff->layers[i])->score_futures;
      groupby_idx.push_back(i);
      float max_i = futures->front().get_result<float>();
      futures->pop_front();
      groupby_max.push_back(max_i);
    }
  }


  // printf("GB: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", groupby_max[0][0], groupby_max[0][1], groupby_max[0][2], groupby_max[0][3], groupby_max[0][4], groupby_max[0][5], groupby_max[0][6], groupby_max[0][7]);

  float new_alpha = -1.0f;
  switch(recompiles)
  {
    case 0:
      // if below 1.58, put 1.8
      if(groupby_max[0] < 1.58f)
        new_alpha = 1.8f;
      break;
    case 1:
      // if above 1.82 put 2.0
      if(groupby_max[0] > 1.82f)
        new_alpha = 2.0f;
      break;
    case 2:
      // if above 2.0 put 2.2
      if(groupby_max[0] > 2.0f)
        new_alpha = 2.2f;
      break;
    case 3:
      // if above 2.2 put 2.5
      if(groupby_max[0] > 2.2f)
        new_alpha = 2.5f;
      break;
    case 4:
      // ir above 2.5 put 2.75
      if(groupby_max[0] > 2.5f)
        new_alpha = 2.75f;
      break;
    case 5:
      // if above 2.8 put 3.0
      if(groupby_max[0] > 2.8f)
        new_alpha = 3.0f;
      break;
    default:
      break;
  }

  if(new_alpha > 0.0f && false) {
    printf("\n\nalter alpha: %.2f\n\n\n", new_alpha);
    ((GroupBy*)ff->layers[groupby_idx[0]])->alpha = new_alpha;

    vector<int> changed_layers;
    changed_layers.push_back(9);
    ff->recompile(changed_layers);
    glob_trace_id++;
    recompiles++;
  }

  return;


  // float cache_thresh = 1.0f;
  // float groupby_thresh_min = 0.6f; //0.6f
  // float groupby_thresh_max = 0.92f; // 1.0f
  // float groupby_overhead_min = 1.25f;
  // float groupby_overhead_max = 1.25f;
  // float max_factor = 4.0f;
  // float cache_score = 0.0f;
  // std::vector<int> groupby_idx;
  // std::vector<float> groupby_max;
  //
  // // get cache score and groupby max
  // // TODO: normalize scores (divide by amt of futures)
  // for(size_t i = 0; i < ff->layers.size(); i++) {
  //   if(ff->layers[i]->op_type == OP_CACHE) {
  //     std::deque<Future>* futures = &((Cache*)ff->layers[i])->score_futures;
  //     // TODO: Here only pop one per partition
  //     while(!futures->empty()) {
  //       cache_score += futures->front().get_result<float>();
  //       futures->pop_front();
  //     }
  //   }
  //   else if(ff->layers[i]->op_type == OP_GROUP_BY) {
  //     std::deque<Future>* futures = &((GroupBy*)ff->layers[i])->score_futures;
  //     groupby_idx.push_back(i);
  //     float max_i = 0.0f;
  //     while(!futures->empty()) {
  //       max_i += futures->front().get_result<float>();
  //       futures->pop_front();
  //     }
  //     groupby_max.push_back(max_i);
  //   }
  // }
  //
  // // alter on condition
  // // if(epoch > 1) {
  //   bool remap = false;
  //   // determine if reallocate alpha
  //   for(size_t i = 0; i < groupby_idx.size(); i++) {
  //     // printf("GB SCORE %d %.3f < %.3f\n", groupby_idx[i], groupby_max[i], ((GroupBy*)ff->layers[groupby_idx[i]])->alpha);
  //     if(groupby_max[i] < groupby_thresh_min*((GroupBy*)ff->layers[groupby_idx[i]])->alpha) {
  //       //((GroupBy*)ff->layers[groupby_idx[i]])->resize_exp_batch(ff, groupby_overhead*groupby_max[i]);
  //       float new_alpha = groupby_overhead_min*groupby_max[i];
  //       printf("\n\nalter alpha: %.3f -> %.3f\n\n", ((GroupBy*)ff->layers[groupby_idx[i]])->alpha, new_alpha);
  //       ((GroupBy*)ff->layers[groupby_idx[i]])->alpha = new_alpha;
  //       ((GroupBy*)ff->layers[groupby_idx[i]])->first_init = true;
  //       vector<int> changed_layers;
  //       changed_layers.push_back(9);
  //       for(int l = 11; l < 115; l++)
  //         changed_layers.push_back(l);
  //
  //       ff->recompile(changed_layers);
  //       glob_trace_id++;
  //     }
  //     else if(((GroupBy*)ff->layers[groupby_idx[i]])->alpha < max_factor && groupby_max[i] > groupby_thresh_max*((GroupBy*)ff->layers[groupby_idx[i]])->alpha) {
  //       float new_alpha = std::min(groupby_overhead_max*groupby_max[i], max_factor);
  //       printf("\n\nalter alpha: %.3f -> %.3f\n\n", ((GroupBy*)ff->layers[groupby_idx[i]])->alpha, new_alpha);
  //       ((GroupBy*)ff->layers[groupby_idx[i]])->alpha = new_alpha;
  //       vector<int> changed_layers;
  //       changed_layers.push_back(9);
  //       for(int l = 11; l < 115; l++)
  //         changed_layers.push_back(l);
  //       ff->recompile(changed_layers);
  //       glob_trace_id++;
  //     }
  //   }
  // // }
  //
  // return;
  //
  // // dermine if cache trigger
  // if(cache_score > cache_thresh && false) {
  //   printf("alter cache!!\n");
  //   ((Cache*)ff->layers[4])->use_cached(true);
  //   // Group by input
  //   ff->layers[5]->inputs[1] = ff->layers[4]->outputs[0];
  //   ff->layers[5]->input_lps[1] = ff->layers[4]->outputs[0].part;
  //   ff->layers[5]->input_grad_lps[1] = ff->layers[4]->outputs[0].part_grad;
  //   // Aggregate input
  //   ff->layers[22]->inputs[1] = ff->layers[4]->outputs[0];
  //   ff->layers[22]->input_lps[1] = ff->layers[4]->outputs[0].part;
  //   ff->layers[22]->input_grad_lps[1] = ff->layers[4]->outputs[0].part_grad;
  //   // AggregateSpec input
  //   ff->layers[23]->inputs[1] = ff->layers[4]->outputs[0];
  //   ff->layers[23]->input_lps[1] = ff->layers[4]->outputs[0].part;
  //   ff->layers[23]->input_grad_lps[1] = ff->layers[4]->outputs[0].part_grad;
  //
  //   remap = true;
  // }

  // return remap;
}
#endif



void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
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

  //float alpha = 3.0f; // factor overhead tensor size for imbalance
  // std::vector<float> alpha = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 4.0f, 0.1f, 4.0f};
  //std::vector<float> alpha = {1.09f, 1.01f, 0.95f, 1.0f, 0.98f, 1.0f, 0.97f, 1.0f};
  // std::vector<float> alpha = {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};

  // std::vector<float> alpha = {2.4f, 2.3f, 1.0f, 1.7f, 1.4f,1.8f, 1.25f, 1.0f};
  float alpha = 3.0f;

  float lambda = 0.03f/100.0f; // 0.06f/250.0f;  // multiplier for load balance term

  // MoE model
#ifdef USE_CNN
  Tensor t = ff.conv2d(input, 64, 11, 11, 4, 4, 2, 2, AC_MODE_RELU);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  t = ff.conv2d(t, 192, 5, 5, 1, 1, 2, 2, AC_MODE_RELU);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  Tensor gate_preds  = ff.flat(t);
  gate_preds = ff.dense(gate_preds, 64, AC_MODE_SIGMOID);
#else
  Tensor gate_preds = input;
#endif
  gate_preds = ff.dense(gate_preds, num_exp, AC_MODE_SIGMOID);
  gate_preds = ff.softmax(gate_preds);

  Tensor topK_output[2];
  ff.top_k(gate_preds, topK_output, num_select, false);
  // ff.cache(topK_output[1], NUM_SAMPLES / ffConfig.batchSize, moe_score);

  Tensor exp_tensors[num_exp];
  ff.group_by(input, topK_output[1], exp_tensors, num_exp, alpha);

  Tensor agg_inputs[num_exp+4];
  agg_inputs[0] = ff.softmax(topK_output[0]); // gate preds
  agg_inputs[1] = topK_output[1]; // gate assign
  agg_inputs[2] = topK_output[1]; // gate assign TopK (for cache)
  agg_inputs[3] = gate_preds; // full gate preds
  for(int i = 0; i < num_exp; i++) {
#ifdef USE_CNN
   Tensor t = ff.conv2d(exp_tensors[i], 64, 11, 11, 4, 4, 2, 2, AC_MODE_RELU);
   t = ff.conv2d(t, 192, 5, 5, 1, 1, 2, 2, AC_MODE_RELU);
    // Tensor t = ff.conv2d(input, 64, 11, 11, 4, 4, 2, 2, AC_MODE_RELU);
    t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
    t = ff.conv2d(t, 192, 5, 5, 1, 1, 2, 2, AC_MODE_RELU);
    t = ff.conv2d(t, 128, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
    t = ff.conv2d(t, 64, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
    t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
    t = ff.flat(t);
    t = ff.dense(t, 128, AC_MODE_RELU/*relu*/);
    t = ff.dense(t, 4096, AC_MODE_RELU/*relu*/);
    t = ff.dense(t, 4096, AC_MODE_RELU/*relu*/);
#else
    Tensor t = exp_tensors[i];
#endif
    Tensor exp_pred = ff.dense(t, OUT_DIM);
    agg_inputs[i+4] = ff.softmax(exp_pred);
    // exp_pred = ff.softmax(exp_pred);
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
  RecompileState r(&moe_trigger, &moe_alter, &ff);
  ff.init_layers();

  // ff.load("j08/c10n5k2b50-fix.ff");
  vector<int> load_layers;
  load_layers.push_back(0);
  load_layers.push_back(1);
  load_layers.push_back(2);
  load_layers.push_back(3);
  ff.load("better_raninit.ff", load_layers);

  // ff.load("j22/profiling.ff");
  // ff.load("j22/c10n5k2-al1-500.ff");



  // // ff.load("c10n5k2b50-shared.ff");
  //
  // int l = 12;
  // for(int i = 0; i < num_exp; i++) {
  //   load_layers.clear();
  //   for(int j = 0; j < 4; j++) {
  //     load_layers.push_back(l);
  //     l++;
  //   }
  //   ff.load("j15/cifar100_backbone_100.ff", load_layers);
  //   // ff.load("j15/c100n16k4-coopbetterraninit.ff", load_layers); //"cifar10_backbone_100.ff"
  //   l += 4;
  // }

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
  for (epoch = 0; epoch < ffConfig.epochs; epoch++) {
    data_loader.reset();
    ff.reset_metrics();
    int iterations = TRAIN_SAMPLES / ffConfig.batchSize;

    for (int iter = 0; iter < iterations; iter++) {
      data_loader.next_batch(ff);
      if (epoch > 0) {
        runtime->begin_trace(ctx, glob_trace_id/*trace_id*/);
      }
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
      if (epoch > 0) {
        runtime->end_trace(ctx, glob_trace_id/*trace_id*/);
      }
      ff.recompile_on_condition(r);
    }

    // // TODO: Do properly
    // ff.reset_metrics();
    // iterations = TEST_SAMPLES / ffConfig.batchSize;
    // for (int iter = 0; iter < iterations; iter++) {
    //   data_loader.next_batch(ff);
    //   ff.forward_test();
    // }

    // if(epoch%10 == 0)
    //   ff.store("j22/profiling.ff");

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
  printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s (comb. train, test)\n", run_time,
         NUM_SAMPLES * ffConfig.epochs / run_time);

  ff.reset_metrics();
  int iterations = TEST_SAMPLES / ffConfig.batchSize;
  for (int iter = 0; iter < iterations; iter++) {
    data_loader.next_batch(ff);
    ff.forward_test();
  }

  // ff.store("jl08/c10n8k2-al3-statwrec.ff");
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
