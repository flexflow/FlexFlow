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

#pragma once

#include "dataloader.h"
#include "inference_config.h"

struct MoeConfig : InferenceConfig {
  MoeConfig(void) : InferenceConfig() {
    //----------------------- MoE layer --------------------------------
    // total number of experts
    num_exp = 64;
    // number of experts in each block of fused experts
    experts_per_block = 16;
    // number of experts to route each token to
    num_select = 2;
    // expert capacity parameters
    alpha = 2.0f;   // factor overhead tensor size for imbalance
    lambda = 0.04f; // multiplier for load balance term
    // expert hidden size
    hidden_size = DATA_DIM;
  }

  // MoE layer
  int num_exp;
  int experts_per_block;
  int num_select;
  float alpha;
  float lambda;
};