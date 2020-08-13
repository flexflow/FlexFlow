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
using namespace Legion;
#define MAX_NPCS 100

LegionRuntime::Logger::Category log_app("AlexNet");

void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
  int npcs = 5, nn_shl[6] = {10, 10, 10, 10, 10, 1};
  FFConfig ffConfig;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    ffConfig.parse_args(argv, argc);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
        ffConfig.batchSize, ffConfig.workersPerNode, ffConfig.numNodes);
  }
  ffConfig.lg_ctx = ctx;
  ffConfig.lg_hlr = runtime;
  ffConfig.field_space = runtime->create_field_space(ctx);
  FFModel ff(ffConfig);
  Tensor pcvec_n, pcvec, pcmax, pcmin;
  {
    const int dims[] = {1, npcs};
    pcvec_n = ff.create_tensor<2>(dims, "", DT_FLOAT);
    pcvec = ff.create_tensor<2>(dims, "", DT_FLOAT);
    pcmax = ff.create_tensor<2>(dims, "", DT_FLOAT);
    pcmin = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  Tensor sb[6];// sb1, sb2, sb3, sb4, sb5
  for (int i = 1; i <= 5; i++) {
    // Treat sb1(:,i) as different tensors for different i
    // to allow additional parallelism across i's
    const int dims[] = {1, nn_shl[i]};
    sb[i] = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  //do i=1,npcs
  //  pcvec_n(i,1)=(((pcvec(i))-pcmin(i,1))/(pcmax(i,1)-pcmin(i,1)))*2.0-1.0
  //enddo
  pcvec_n = ff.divide("", ff.subtract("", pcvec, pcmin), ff.subtract("", pcmax, pcmin));
  Tensor output[MAX_NPCS];
  //do i = 1, npcs
  for (int pc = 1; pc <= npcs; pc++) {
    Tensor s_layer[6]; // s0_layer = pcvec_n, s1_layer, s2_layer, s3_layer, s4_layer
    // We denote s5_layer = output so that we can reuse the for loop
    // to perform the math for output
    s_layer[0] = pcvec_n;
    for (int i = 1; i <= 5; i++) {
      //s1_layer = matmul(sw1(:,:,i),pcvec_n)
      s_layer[i] = ff.dense("", s_layer[i-1], nn_shl[i]);
      // create a constant tensosrs: one, two, and minus_two
      // TODO: a potential optimization is to move the creations out of the loop
      // to avoid multiple creations
      Tensor one, two, minus_two;
      {
        const int dims[] = {1, nn_shl[i]};
        one = ff.create_constant<2>(dims, "", 1, DT_FLOAT);
        two = ff.create_constant<2>(dims, "", 2, DT_FLOAT);
        minus_two = ff.create_constant<2>(dims, "", -2, DT_FLOAT);
      }
      //s1_layer(:,1)=s1_layer(:,1)+sb1(:,i)
      s_layer[i] = ff.add("", s_layer[i], sb[i]);
      // s1_layer(:,1)=1+exp((-2.0)*s1_layer(:,1)
      s_layer[i] = ff.add("", one, ff.exp("", ff.multiply("", minus_two, s_layer[i])));
      //s1_layer(:,1) = (2.0/s_layer[i]-1.0)
      s_layer[i] = ff.subtract("", ff.divide("", two, s_layer[i]), one);
    }
    // Finally output = s5_layer
    output[pc] = s_layer[5];
  }
  // outlayer(i,1) = output(1,1)
  // This is identical to concatenating output[i] along the zero-th dim
  Tensor outlayer = ff.concat("", npcs/*num*/, output+1, 0/*axis*/);
}

void register_custom_tasks()
{}
