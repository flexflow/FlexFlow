/* Copyright 2021 Stanford University
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
#include <fstream>
#include <sstream>
#include <string>
using namespace Legion;
using namespace FlexFlow;

// embed_dim=768,
// num_heads=12,
// kdim=None,
// vdim=None,
// dropout=0.1,
// bias=True,
// add_bias_kv=False,
// add_zero_attn=False,
// self_attention=True,
// encoder_decoder_attention=False,
// q_noise=0.0,
// qn_block_size=8,

// Tensor FFModel::multihead_attention(const Tensor query,
// const Tensor key,
// const Tensor value,
// int embed_dim,
// int num_heads,
// int kdim,
// int vdim,
// float dropout,
// bool bias,
// bool add_bias_kv,
// bool add_zero_attn,
// Initializer *kernel_initializer,
// char const *name) {


void create_attention_decoder(FFModel *model,
                                      Tensor const &input1,
                                      Tensor const &input2,
                                      Tensor &output1,
                                      Tensor &output2,
                                      int embed_dim,
                                      int num_heads,
                                      int kdim,
                                      int vdim,
                                      float dropout=0.1,
                                      bool normalize_before,
                                      bool is_moe) {
  
  std::vector<int> axes = {embed_dim};
  Tensor x = normalize_before ? model->LayerNorm(input1 /*const Tensor input*/, &axes /*std::vector<int> const &axes*/, true /*elementwise_affine*/, 1e-05 /*eps*/) : input1;
  x = model->add(model->dropout(model->multihead_attention(x, x, x, embed_dim, num_heads, embed_dim, embed_dim, dropout, true /*bias*/, false /*add_bias_kv*/, false /*add_zero_attn*/), dropout), x);
  //x = normalize_before ? x : model->LayerNorm(x, &axes, true, 1e-05);
  x = model->LayerNorm(x, &axes, true, 1e-05);

  if(!is_moe) {
    x = model->dropout(model->dense(model->dropout(model->dense(x, 3072, AC_MODE_GELU, true /*bias*/), dropout), embed_dim, AC_MODE_NONE, true /*bias*/), dropout);
  } else {
    // x - seq_len, batch_size, model_dim
    // x = x.transpose(0, 1) # batch_size, seq_len, model_dim
    // x, l_aux = self.moe_layer(x)
    // x = x.transpose(0, 1) # seq_len, batch_size, model_dim
    //x = self.residual_connection(x, residual)
    
    //if not self.normalize_before:
    //    x = self.final_layer_norm(x)
    x = normalize_before ? x : model->LayerNorm(x, &axes, true, 1e-05);
    float alpha = 2.0f;   // factor overhead tensor size for imbalance
    float lambda = 0.04f; // multiplier for load balance term

    // MoE model
    Tensor gate_preds = ff.dense(x, num_exp, AC_MODE_RELU);
    Tensor topK_output[2];
    ff.top_k(gate_preds, topK_output, num_select, false);

    Tensor exp_tensors[num_exp];
    ff.group_by(input, topK_output[1], exp_tensors, num_exp, alpha);

    Tensor agg_inputs[num_exp + 4];
    agg_inputs[0] = ff.softmax(topK_output[0]); // gate preds
    agg_inputs[1] = topK_output[1];             // gate assign
    agg_inputs[2] = topK_output[1];             // gate assign TopK (for cache)
    agg_inputs[3] = gate_preds;                 // full gate preds
    for (int i = 0; i < num_exp; i++) {
      Tensor exp_pred = ff.dense(exp_tensors[i], OUT_DIM, AC_MODE_RELU);
      agg_inputs[i + 4] = ff.softmax(exp_pred);
    }
  }
  
  Tensor t1 =
      model->add(model->multihead_attention(
                     input1, input1, input1, hidden_dim, num_heads, kdim, vdim),
                 input1);
  t1 = model->dense(model->dense(t1, hidden_dim, AC_MODE_RELU, false /*bias*/),
                    hidden_dim,
                    AC_MODE_NONE,
                    false /*bias*/);
  Tensor t2 =
      model->add(model->multihead_attention(
                     input2, input2, input2, hidden_dim, num_heads, kdim, vdim),
                 input2);
  t2 = model->add(
      model->multihead_attention(t2, t1, t1, hidden_dim, num_heads, kdim, vdim),
      t2);
  t2 = model->dense(model->dense(t2, hidden_dim, AC_MODE_RELU, false /*bias*/),
                    hidden_dim,
                    AC_MODE_NONE,
                    false /*bias*/);
  output1 = t1;
  output2 = t2;
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffConfig;
  fprintf(stderr,
          "batchSize(%d) workersPerNodes(%d) numNodes(%d)\n",
          ffConfig.batchSize,
          ffConfig.workersPerNode,
          ffConfig.numNodes);
  FFModel ff(ffConfig);

  std::vector<int> hidden_dims = {
      8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192};
  Tensor input1, input2;
  {
    int const dims[] = {ffConfig.batchSize, 1024};
    input1 = ff.create_tensor<2>(dims, DT_FLOAT);
    input2 = ff.create_tensor<2>(dims, DT_FLOAT);
  }
  Tensor t1 = input1, t2 = input2;
  for (size_t i = 0; i < hidden_dims.size(); i++) {
    int const dims[] = {hidden_dims[i], t1->dims[0]};
    ActiMode acti_mode =
        (i + 1 == hidden_dims.size()) ? AC_MODE_NONE : AC_MODE_RELU;
    t1 = ff.dense(t1, hidden_dims[i], acti_mode, false);
    t2 = ff.dense(t2, hidden_dims[i], acti_mode, false);
  }
  Tensor t = ff.add(t1, t2);
  t = ff.softmax(t);
  Optimizer *optimizer = new SGDOptimizer(&ff, 0.001f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics, CompMode::COMP_MODE_INFERENCE);
  ff.init_operators();
  // Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();
  //for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
  ff.reset_metrics();
  int iterations = 128;
  for (int iter = 0; iter < iterations; iter++) {
    runtime->begin_trace(ctx, 111 /*trace_id*/);
    ff.forward();
    runtime->end_trace(ctx, 111 /*trace_id*/);
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
         ffConfig.batchSize * 128 * ffConfig.epochs / run_time);
}

void FlexFlow::register_custom_tasks() {}
