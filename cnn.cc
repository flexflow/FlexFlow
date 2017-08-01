/* Copyright 2017 Stanford, NVIDIA
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

#include <cstdio>
#include "legion.h"
#include "ops.h"

using namespace Legion;

LegionRuntime::Logger::Category log_cnn("cnn");

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  // Create a CnnConfig
  CnnConfig config;
  config.lg_ctx = ctx;
  config.lg_hlr = runtime;
  config.num_par_h = 2;
  config.num_par_w = 2;
  config.num_par_n = 1;
  int num_workers = config.num_par_h * config.num_par_w * config.num_par_n;
  // First, create cnnContexts
  std::vector<CnnHandle*> contexts;
  for (int i = 0; i < num_workers; i++) {
    contexts.push_back(new CnnHandle());
  }
  Realm::ZRect<1, coord_t> init_rect(0, num_workers - 1);
  IndexSpaceT<1> init_is = runtime->create_index_space(ctx, init_rect);
  ArgumentMap local_args;
  size_t workSpaceSize = (size_t)4 * 1024 * 1024 * 1024;
  IndexLauncher init_launcher(INIT_TASK_ID, init_is,
                              TaskArgument(&workSpaceSize, sizeof(workSpaceSize)), local_args);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  for (int idx = 0; idx < num_workers; idx++) {
    contexts[idx] = fm.get_result<CnnHandle*>(idx);
  }

  // Initialize every layer
  for (int i = 0; i < num_workers; i++) {
    
  }
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_TASK_ID, "init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<CnnHandle, init_cudnn>(registrar, "init_task");
  }

  return Runtime::start(argc, argv);
}
