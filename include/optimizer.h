/* Copyright 2019 Stanford
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

#ifndef _FF_OPTIMIZER_H_
#define _FF_OPTIMIZER_H_

#include "legion.h"

using namespace Legion;

class FFModel;
class Parameter;

class Optimizer
{
public:
  Optimizer(const FFModel* _model);
  virtual void next(void) = 0;
  virtual void update(const Parameter* p) = 0;
  const FFModel* model;
};

class SGDOptimizer : public Optimizer
{
public:
  SGDOptimizer(const FFModel* _model,
               double lr = 0.01f, double momentum = 0.0f,
               bool nesterov = false, double weight_decay = 0.0f);
  void next(void);
  void update(const Parameter* p);
  void set_weight_decay(double _weight_decay);
  static void update_task(const Task* task,
                          const std::vector<PhysicalRegion>& regions,
                          Context ctx, Runtime* runtime);
  double lr, momentum;
  bool nesterov;
  double weight_decay;
  std::map<LogicalRegion, LogicalRegion> v_regions;
  std::map<LogicalRegion, LogicalPartition> v_parts;
};
#endif
