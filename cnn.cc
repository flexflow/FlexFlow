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

using namespace LegionRuntime::HighLevel;

LegionRuntime::Logger::Category log_cnn("cnn");

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  // First, create cnnContexts
  std::vector<CnnContext*> contexts;
  for (int i = 0; i < num_workers; i++) {
    contexts.push_back(new CnnContext());
  }
  Rect<1> init_rect(0, num_workers - 1);
  IndexSpaceT<1> init_is = runtime->create_index_space(ctx, init_rect);
  ArgumentMap local_args;
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, init_is,
                              TaskArgument(NULL, 0), local_args);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  for (int idx = 0; idx < num_workers; idx++) {
    contexts[idx]->handle = fm.get_result<CnnHandle>(idx);
  }

  // Initialize every layer
  for (int i = 0; i < num_workers; i++) {
    
  }
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
    Processor::LOC_PROC, true/*single*/, false/*index*/,
    CNN_CPU_LEAF_VARIANT, TaskConfigOptions(), "top_level");

  return HighLevelRuntime::start(argc, argv);
}
