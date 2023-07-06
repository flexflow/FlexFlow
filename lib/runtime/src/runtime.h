#ifndef _FLEXFLOW_RUNTIME_SRC_RUNTIME_H
#define _FLEXFLOW_RUNTIME_SRC_RUNTIME_H

#include "utils/visitable.h"
#include "parallel_computation_graph.h"
#include "legion_backing.h"

namespace FlexFlow {

struct ModelTrainingInstance {
public:
  ModelTrainingInstance() = delete;
  
  void execute(TaskInvocation const &) const;
  void execute(OpTaskInvocation const &) const;

public:
  ParallelComputationGraph pcg;
  RuntimeBacking backing; 
};

void forward(ModelInstance const &);
void backward(ModelInstance const &);


}

VISITABLE_STRUCT(::FlexFlow::ModelInstance);

#endif
