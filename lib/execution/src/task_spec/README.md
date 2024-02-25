# task\_spec

The `task_spec` interface provides an easy-to-use, high-level, and safe abstraction on top of Legion tasks.
While not all Legion features are supported, the `task_spec` interface is capable of expressing all Legion usages in FlexFlow.
Using `task_spec` is not mandatory (Legion still works fine, as everything simply compiles down to Legion `TaskLauncher`, etc. 
anyway), but any code that can use `task_spec` is strongly advised to use it as it is significantly less verbose, safer, and 
prevents common errors.

The `task_spec` code consists of two parts: `TaskSignature` ([task\_signature.h](./task_signature.h)) and `TaskInvocation` ([task\_invocation.h](./task_invocation.h)), 
which can be intuitively understood as function signatures and function calls in a typical programming language.
`TaskSignature`s define a set of _slots_ of two kinds: 
each can be either a _tensor slot_, which represents a parallel tensor whose Legion region will be passed to the underlying task, 
or an _argument slot_, which can be used to pass small[^1] values of arbitrary[^2] type via `Legion::TaskArgument`.

As with function signatures/calls, each task has a single `TaskSignature` but can have multiple `TaskInvocation`s.
`TaskSignature`s are registered for `task_id_t`s via the `register_task` function, which is usually called by specializations of `template <task_id_t> register_task` 
defined in the relevant file (e.g., [optimizer.h](../optimizer.h) and [optimizer.cc](../optimizer.cc)), which are ultimately called by 
`register_flexflow_internal_tasks` in [tasks.cc](../tasks.cc).

To execute a pair of a `TaskSignature` and a `TaskInvocation`, they must be compiled/translated/lowered to a call to a `Legion::TaskLauncher` or a 
`Legion::IndexTaskLauncher`.
Ideally this would simply be done in a single step, but in practice the ability to specify `TaskInvocation`s at different layers of abstraction can 
be very useful.
Thus, what we previously referred to as `TaskInvocation` is actually logically the following set of classes:

```mermaid
flowchart TD
    A[OpTaskInvocation]
    B[TaskInvocation]
    C[ExecutableTaskInvocation]
    D[TensorlessTaskInvocation]
    E[IndexTaskInvocation]
    F[Legion::TaskLauncher]
    G[Legion::IndexTaskLauncher]
    H[ExecutableIndexTaskInvocation]
    I[TensorlessIndexTaskInvocation]
    A -->|compiles down to| E
    E -->|compiles down to| H
    H -->|compiles down to| I 
    I -->|compiles down to| G

    B -->|compiles down to| C
    C -->|compiles down to| D
    D -->|compiles down to| F
```
Similarly, `TaskSignature` is actually divided up into `OpTaskSignature` and `TaskSignature`.
The flow of full compilation process is as follows:
```mermaid
%%{init: { 'themeVariables': { 'fontFamily': 'monospace' }, 'flowchart': { 'curve': 'bumpY', 'defaultRenderer': 'elk' }, 'theme': 'default' } }%%
flowchart TD
    A[OpTaskInvocation]
    B[TaskInvocation]
    C[ExecutableTaskInvocation]
    D[TensorlessTaskInvocation]
    E[IndexTaskInvocation]
    F[Legion::TaskLauncher]
    G[Legion::IndexTaskLauncher]
    H[ExecutableIndexTaskInvocation]
    I[TensorlessIndexTaskInvocation]
    J[OpTaskSignature]
    K[TaskSignature]
    L[ConcreteArgsFormat]
    M[FutureArgsFormat]
    N[TensorArgsFormat]
    O[IndexArgsFormat]
    P[TaskArgumentsFormat]
    Q[Legion::TaskArgument]
    R[Legion::ArgumentMap]
    S[TaskReturnAccessor]
    T[IndexTaskReturnAccessor]
    AA[task_id_t]
    AC[TensorlessTaskBinding]
    AD[TensorlessIndexTaskBinding]
    AE[task_impl function]
    AF[task function]
    AG[Legion::Task]
    AH["std::vector<Legion::PhysicalRegion>"]
    AI[Legion::Context]
    AJ[Legion::Runtime]
    AK[TaskArgumentAccessor]
    AL[add_region_requirement]

    A -->|compiles to| E
    E -->|compiles to| H
    H -->|compiles to| N
    N -->|compiles to| P
    N -->|invokes| AL
    AL -->|on| G
    H -->|compiles to| I
    I -->|has member| AA
    I -->|has member| AD 
    AD -->|compiles to| M
    AD -->|compiles to| O
    AD -->|compiles to| L
    O -->|compiles to| R
    O -->|compiles to| P
    M -->|compiles to| P
    L -->|compiles to| P
    M -->|compiles to| Q
    O -->|compiles to| Q
    L -->|compiles to| Q
    P -->|compiles to| Q
    Q -->|passed to| G
    R -->|passed to| G
    G -->|generates a| AG
    G -->|generates a| AH
    G -->|generates a| AI 
    G -->|generates a| AJ
    AG -->|passed to| AF
    AH -->|passed to| AF
    AI -->|passed to| AF 
    AJ -->|passed to| AF
    AF -->|generates a| AK
    AK -->|passed to| AE
    AE -->|possibly generates a| S
    G -->|possibly generates a| S
    K -->|possibly generates a| S

    B -->|compiles to| C
    C -->|compiles to| N
    C -->|compiles to| D
    D -->|has member| AA
    D -->|has member| AC
    AC -->|compiles to| L 
    AC -->|compiles to| M
    L -->|compiles to| P
    M -->|compiles to| P 
    L -->|compiles to| Q
    M -->|compiles to| Q
    P -->|compiles to| Q
    Q -->|passed to| F
    AL -->|on| F
    F -->|generates a| AG
    F -->|generates a| AH
    F -->|generates a| AI 
    F -->|generates a| AJ
    AE -->|possibly generates a| T
    G -->|possibly generates a| T
    K -->|possibly generates a| T

    J -->|compiles to| K
```

The primary difference between the different `TaskInvocation` types is which argument types they support.
The full list of argument types is:
- tensor slots
  - `OpTensorSpec`: a reference to a input, output, or weight tensor attched to the given operator. 
  - `ParallelTensorSpec`: a reference (via `parallel_tensor_guid_t`) to a parallel tensor somewhere in the PCG.
- argument slots
  - `OpArgRefSpec`: an argument that should be filled in during the compilation process from `OpTaskInvocation` to `TaskInvocation`. For those familiar with `Reader` monads, this is roughly analogous
  - `ConcreteArgSpec`: a concrete value
  - `IndexArgSpec`: a set of concrete values, each of which should be sent to a different Index Task
  - `CheckedTypedFuture`: a legion future whose value should be passed into the task
  - `CheckedTypedFutureMap`: a set of legion futures, each of which should have its value sent to a different Index Task (conceptually, `IndexArgSpec` + `CheckedTypedFuture`)
  - `ArgRefSpec`: an argument that should be filled in during the compilation process from `TaskInvocation` to `ExecutableTaskInvocation`. For those familiar with `Reader` monads, this is roughly analogous
  - `TaskInvocationSpec`: a nested task invocation which should be launched and have its resulting `Future` passed into the given task
  - `IndexTaskInvocationSpec`: (currently not implemented, may or may not be necessary)

The supported argument types for each invocation type are:
- `OpTaskInvocation`
  - `OpTensorSpec`, `OpArgRefSpec`, `ConcreteArgSpec`, `IndexArgSpec`, `CheckedTypedFuture`, `CheckedTypedFutureMap`, `ArgRefSpec`, `TaskInvocationSpec`, `IndexTaskInvocationSpec`
- `TaskInvocation`
  - `ParallelTensorSpec`, `ConcreteArgSpec`, `CheckedTypedFuture`, `ArgRefSpec`, `TaskInvocationSpec`
- `IndexTaskInvocation`
  - `ParallelTensorSpec`, `ConcreteArgSpec`, `IndexArgSpec`, `CheckedTypedFuture`, `CheckedTypedFutureMap`, `ArgRefSpec`, `TaskInvocationSpec`, `IndexTaskInvocationSpec`
- `ExecutableTaskInvocation`
  - `ParallelTensorSpec`, `ConcreteArgSpec`, `CheckedTypedFuture`, `TaskInvocationSpec`
- `ExecutableIndexTaskInvocation`
  - `ParallelTensorSpec`, `ConcreteArgSpec`, `IndexArgSpec`, `CheckedTypedFuture`, `CheckedTypedFutureMap`, `TaskInvocationSpec`, `IndexTaskInvocationSpec`
- `TensorlessTaskInvocation`
  - `ConcreteArgSpec`, `CheckedTypedFuture`, `TaskInvocationSpec`
- `TensorlessIndexTaskInvocation`
  - `ConcreteArgSpec`, `IndexArgSpec`, `CheckedTypedFuture`, `CheckedTypedFutureMap`, `TaskInvocationSpec`, `IndexTaskInvocationSpec`

[^1]: i.e., not tensor-sized
[^2]: Types must either be serializable ([serialization.h](../serialization.h)) or device-specific ([device\_specific\_arg.h](./device-specific-arg.h))
