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

#ifndef _FF_RECOMPILE_H_
#define _FF_RECOMPILE_H_

#include "legion.h"
#include <functional>

using namespace Legion;


class FFModel;

class RecompileState
{
public:
  RecompileState(FFModel* ff, std::function<bool(FFModel*, RecompileState&)> _alter_func,
               int _launch_ahead=1);
public:
  int recompilations;
  int last_recompile;
  std::function<bool(FFModel*, RecompileState&)> alter_func;
  // FFModel* ff;
  int launch_ahead;
};
#endif
