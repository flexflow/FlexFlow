/* Copyright 2020 Stanford University, Los Alamos National Laboratory
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

#include "legion.h"

using namespace Legion;

#include "mapper.h"
#include "flexflow_c.h"

void register_flexflow(int argc, char **argv)
{
#ifdef FF_USE_NCCL
  // Set NCCL environment
  // This needs to be set, otherwise NCCL will try to use group kernel launches,
  // which are not compatible with the Realm CUDA hijack.
  setenv("NCCL_LAUNCH_MODE", "PARALLEL", true);
#endif

  register_flexflow_internal_tasks();

  register_c_custom_tasks();

  FFMapper::register_sharding_functor(argc, argv);

  Runtime::add_registration_callback(update_mappers);
}
