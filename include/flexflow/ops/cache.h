#ifndef _FLEXFLOW_CACHE_H_
#define _FLEXFLOW_CACHE_H_

#include "flexflow/model.h"

namespace FlexFlow {

class Cache;

class CacheMeta : public OpMeta {
public:
  CacheMeta(FFHandler handle, Cache const *c);
  float cache_score;
};

class Cache : public Op {
public:
  Cache(
      FFModel &model,
      ParallelTensor const &_input,
      int _num_batches,
      std::function<float(float *, void const *, void const *, int)> &_score_f,
      char const *name);
  ~Cache(void);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  // void create_weights(FFModel& model);
  // void create_output_and_partition(FFModel& model);

  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static float update_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  template <typename T>
  static void cache_forward(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <typename T>
  static float cache_update(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  void use_cached(bool cached);

public:
  void **batch_ptrs;
  void *batch_cmp;
  bool load_cached;
  int num_batches;
  std::function<float(float *, void const *, void const *, int)> score_f;
  std::vector<Legion::Future> score_futures;
  int batch_ctr;
};

struct Arg {
  Cache *cache;
  int batch_ctr;
};

}; // namespace FlexFlow

#endif
