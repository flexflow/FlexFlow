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

#include "mapper.h"
#include "model.h"

using namespace Legion;
using namespace FlexFlow;

// ========================================================
// Task and mapper registrations
// ========================================================
int main(int argc, char **argv) {
  // This needs to be set, otherwise NCCL will try to use group kernel launches,
  // which are not compatible with the Realm CUDA hijack.
  setenv("NCCL_LAUNCH_MODE", "PARALLEL", true);

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  register_flexflow_internal_tasks();

  // Register custom tasks
  register_custom_tasks();

  Runtime::add_registration_callback(FFMapper::update_mappers);
  return Runtime::start(argc, argv);
}
