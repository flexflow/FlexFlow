# Task Spec DSL

There are three components in task-based systems: `agent`, `task`, and `backend`. At a high level, they interact with each other in the following way: an `agent` defines a `task` which is executed by the `backend`. In FlexFlow, an `agent` could be an operator (e.g., `Conv2D`), a `task` could be its `forward()` function, and this could be executed on a `backend` like the Legion runtime. A more complicated example is a deep neural network, in which there are many `agent`s dispatching many `task`s that may depend on each other. In this document, we will define the three terms and their interactions.

## Definitions

The generic use case for our system is the following:
1. Allocate a memory region
2. Read from this region
3. Apply a transformation
4. Write the result of the transformation back to memory
5. De-allocate the memory region

We can separate this into three layers. The highest layer is the `agent`, which defines the transformation and memory requirements. The middle layer is the `task`, which is responsible for translating the `agent`'s request into memory locations (via a lookup table) and applying the transformation. The lowest layer is the `backend`, which will retrieve and modify memory at these locations (in addition to being able to allocate and de-allocate memory).

## Sequencing

Typically, we will need to complete multiple tasks in sequence and possibly modify the same memory locations. For example, there may be an `Embedding` operator, whose output should pass to the `Linear` operator. Hence, we can define a `task` more clearly:
- A set of inputs $X$
- A composition of transformations $F$, each applied in sequence
- A set of outputs $Y$

It will lookup the inputs in its table, fetch relevant memory regions, apply its transformation, write to relevant memory regions, and write the output locations in its table. 

To complete this definition, let's introduce two ideas: 
1. `taskGraph` - a DAG that defines the interactions between `task`s
2. `future` - a pointer to the index of another `task`'s lookup table. This must follow the direction established in a `taskGraph` in that the `task` being pointed *to* must be sequentially prior to the `task` being pointed *from*. 

To handle asynchronous execution, even if the result of a future is unavailable, a `task`'s transformations can still be theoretically applied. Its output will aggregate the transformations that need to be applied and then execute those transformations once the result is available. 


## Composition

Given two `task` in a sequence, our representation allows composition into a single `task`. Let $T_1$ and $T_2$ represent our sequential `task`s. The result of executing $T_2 \circ T_1$ is $F_2(F_1(x_1))$. $T_2 \circ T_1$ itself is a `task` because it meets the definition of a task: its input is $x_1$, its transformation is $F_2 \circ F_1$, and the output is $y_2$.

A more complicated example. Let $T_1$ and $T_2$ represent our sequential `task`s. In this case, $T_2$ has inputs $\{x_2, x_3\}$ where only $x_2$ holds a future to $F_1(x_1)$. Hence, $T_2 \circ T_1$ is $F_2(F_1(x_1), x_3)$. Still, it is a task, only with inputs: $x_1, x_3$. Conveniently, this is $X_1 \cup (X_2 \cap Y_1'))$.

This composition gives us a useful tactic: the ability to aggregate `task`s into a single `task`. The inputs for the aggregated `task` are the union of the inputs to all the `task`s that are **not** intermediate results. The output is the output of the final `task`. 

There are two rules we have to follow when aggregating:
1. Source rule: If the inputs of a `task` $t$ is are `future`s to more than one `task`, $t$ can only be the first `task` in its aggregation chain.
2. Sink rule: If an output of a `task` $t$ is referenced by `future`s in more than one `task`, $t$ will be the final `task` in its aggregation chain. 
   
Basically, aggregation must stop at branches in the `taskGraph`.  

## `Legion::Future`

In `Metrics::compute`
```
  FutureMap new_metrics = runtime->execute_index_space(ctx, launcher);
  // Update metrics
  TaskLauncher metrics_task(UPDATE_METRICS_TASK_ID,
                            TaskArgument(this, sizeof(Metrics)));
  metrics_task.add_future(model->current_metrics);
  for (Domain::DomainPointIterator it(part_domain); it; it++) {
    metrics_task.add_future(new_metrics[*it]);
  }
  model->current_metrics = runtime->execute_task(ctx, metrics_task);
```

In `BatchNorm::backward` it’s called without `wait_all_results()` (but nothing is ever done with the future map)
- Same for Conv2D

In `Cache` the futures are stored in an `std::vector<Legion::Future> score_futures` but this is only utilized in the MOE examples, nowhere else.

In `optimizer.cc`, `SGDOptimizer::update` and `AdamOptimizer::update` have this code
```
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    // runtime->execute_must_epoch(ctx, must_epoch_launcher);
    runtime->issue_execution_fence(ctx);
```
But `fm` is never referenced

The `inference` branch uses futures everywhere
```
  launcher.add_future(bc);
  runtime->execute_index_space(ctx, launcher);
```
