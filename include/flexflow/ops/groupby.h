#ifndef _FLEXFLOW_GROUPBY_H_
#define _FLEXFLOW_GROUPBY_H_

#include "flexflow/model.h"

namespace FlexFlow {

class GroupByMeta : public OpMeta {
public:
  GroupByMeta(FFHandler handle, int n);
  ~GroupByMeta(void);
  float** dev_region_ptrs;
};

class Group_by : public Op {
public:
  Group_by(FFModel& model,
          const Tensor _input,
          const Tensor _assign,
          int _n, float _alpha,
          const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
public:
  int n;
  float alpha;
};

}; // namespace FlexFlow

#endif
