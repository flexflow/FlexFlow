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

RecompileState::RecompileState(std::function<bool(FFModel*)> _trigger_func,
                              std::function<void(FFModel*)> _alter_func,
                              FFModel* _ff)
: trigger_func(_trigger_func), alter_func(_alter_func), ff(_ff)
{
  recompilations = 0;
}

bool RecompileState::trigger() {
  return trigger_func(ff);
}

void RecompileState::alter() {
  if(recompilations == 0)
    alter_func(ff);
  recompilations++;
}
