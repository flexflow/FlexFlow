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

## Composition

Given two `task` in a sequence, our representation allows composition into a single `task`. Let $T_1$ and $T_2$ represent our sequential `task`s. The result of executing $T_2 \circ T_1$ is $F_2(F_1(x_1))$. $T_2 \circ T_1$ itself is a `task` because it meets the definition of a task: its input is $x_1$, its transformation is $F_2 \circ F_1$, and the output is $y_2$.

A more complicated example. Let $T_1$ and $T_2$ represent our sequential `task`s. In this case, $T_2$ has inputs $\{x_2, x_3\}$ where only $x_2$ holds a future to $F_1(x_1)$. Hence, $T_2 \circ T_1$ is $F_2(F_1(x_1), x_3)$. Still, it is a task, only with inputs: $x_1, x_3$. Conveniently, this is $X_1 \cup (X_2 \cap Y_1'))$.

This composition gives us a useful tactic: the ability to aggregate `task`s into a single `task`. The inputs for the aggregated `task` are the union of the inputs to all the `task`s that are **not** intermediate results. The output is the output of the final `task`. The only rule we have to follow when aggregating is that if an output of a `task` $t$ is referenced by `future`s in more than one `task`, $t$ will be the final `task` in its aggregation chain. Basically, aggregation must stop at branches in the `taskGraph`.  