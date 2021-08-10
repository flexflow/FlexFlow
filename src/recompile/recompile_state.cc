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
#include "recompile.h"
#include "legion.h"

using namespace Legion;

// TODO: Du brauchst wahrsch die parallel config
// TODO: q_len muss mal anzahl devices etc sein

RecompileState::RecompileState(FFModel* ff, std::function<bool(FFModel*, RecompileState&)> _alter_func,
                              size_t _launch_ahead)
: alter_func(_alter_func), launch_ahead(_launch_ahead)
{
  assert(launch_ahead > 0); 

  // some statistics to use in the alter function
  recompilations = 0;
  last_recompile = 0;

  // set q_len for all operators with a score
  for(int l = 0; l < ff->layers.size(); l++) {
    switch(ff->layers[l]->op_type) {
      case OP_GROUP_BY:
        ((GroupBy*)ff->layers[l])->q_len = launch_ahead;
        break;
      case OP_CACHE:
        ((Cache*)ff->layers[l])->q_len = launch_ahead;
        break;
      default:
        break;
    }
  }
}

// void RecompileState::alter(bool perform_rec)
// {
//   last_recompile++;
//
//   if(q_len < launch_ahead) return;
//
//   bool rec = alter_func(ff, this);
//
//   if(rec && perf_rec) {
//     // TODO: Search for parallelization strategy
//     ff->recompile();
//     recompilations++;
//     last_recompile = 0;
//   }
// }
