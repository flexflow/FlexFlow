/* Copyright 2023 CMU, Facebook, Stanford
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

using namespace FlexFlow;

Tensor FFModel::moe(const Tensor input,
                    int num_exp,
                    int num_select,
                    int expert_hidden_size,
                    float alpha,
                    float lambda) {
  // MoE model
  Tensor gate_preds = dense(input, num_exp, AC_MODE_RELU);
  Tensor topK_output[2];
  top_k(gate_preds, topK_output, num_select, false);
  Tensor exp_tensors[num_exp];
  group_by(input, topK_output[1], exp_tensors, num_exp, alpha);
  Tensor agg_inputs[num_exp + 4];
  agg_inputs[0] = softmax(topK_output[0]); // gate preds
  agg_inputs[1] = topK_output[1];          // gate assign
  agg_inputs[2] = topK_output[1];          // gate assign TopK (for cache)
  agg_inputs[3] = gate_preds;              // full gate preds
  for (int i = 0; i < num_exp; i++) {
    Tensor exp_pred = dense(exp_tensors[i], expert_hidden_size, AC_MODE_RELU);
    agg_inputs[i + 4] = softmax(exp_pred);
  }
  Tensor coop_output = aggregate(agg_inputs, num_exp, lambda);
  // get_metrics();
  return coop_output;
}
