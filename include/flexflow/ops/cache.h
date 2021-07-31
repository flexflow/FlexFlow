#ifndef _FLEXFLOW_CACHE_H_
#define _FLEXFLOW_CACHE_H_

#include "flexflow/model.h"

namespace FlexFlow {

class CacheMeta : public OpMeta {
public:
  CacheMeta(FFHandler handle);
  float cache_score;
};

class Cache : public Op {
public:
  Cache(FFModel& model,
      const Tensor& _input,
      int _num_batches,
      std::function<float(float*,const void*,const void*,int)> &_score_f,
      const char* name);
  ~Cache(void);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static float update_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  void use_cached(bool cached);
public:
  void** batch_ptrs;
  void* batch_cmp;
  bool load_cached;
  int num_batches;
  std::function<float(float*,const void*,const void*,int)> score_f;
  std::vector<Legion::Future> score_futures;
  int batch_ctr;
};

struct Arg {
  Cache* cache;
  int batch_ctr;
};

}; // namespace FlexFlow

#endif
