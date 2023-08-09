#ifndef _FLEXFLOW_RUNTIME_SRC_RUNTIME_H
#define _FLEXFLOW_RUNTIME_SRC_RUNTIME_H

#include "legion_backing.h"
#include "parallel_computation_graph.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ModelInstance {
public:
  ModelInstance() = delete;

  void execute(TaskInvocation const &) const;
  void execute(OpTaskInvocation const &) const;

public:
  ParallelComputationGraph pcg;
  RuntimeBacking backing;
};

void forward(ModelInstance const &);
void backward(ModelInstance const &);

TaskInvocation forward(ParallelComputationGraph const &, operator_guid_t);
TaskInvocation backward(ParallelComputationGraph const &, operator_guid_t);

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::ModelInstance);

#endif
