# C++ Library Code

This directory contains the core C++ code that underlies FlexFlow, organized into the following libraries:

- `compiler`: Contains 
- `kernels`:
- `op-attrs`:
- `pcg`: Contains the definitions of computation graphs and parallel computation graphs,
         as well as code for serializing and deserializing both graphs
- `runtime`:
- `substitutions`: Contains the definitions of pcg substitutions, as well as the code for serializing them
- `utils`:

```mermaid
%%{init: {"fontFamily": "monospace"}}%%
sequenceDiagram
    participant U as User
    participant P as ../python
    participant CFFI as C FFI
    participant R as runtime
    participant C as compiler
    participant I as ModelTrainingInstance
    U-->>P: model.compile(algorithm=..., optimizer=...)
    P-->>CFFI: flexflow_computation_graph_t
    CFFI->R: ComputationGraph::conv_2d, ComputationGraph::embedding, etc.
    R-->>C: ComputationGraph + MachineConfiguration + CostEstimator
    C-->>R: ParallelComputationGraph, TensorMapping
    R-->>P: ModelCompilationResult
    P-->>U: ModelCompilationResult
    U-->>P: compiled_model.fit(loss_type=..., metrics=..., enable_profiling=..., x=dataloader_input, y=dataloader_label, epochs=...)
    P-->>R:
    R-->>I:
    opt Reading tensor elements
        U-->>P: get_tensor
        P-->>R:
        R-->>I:
        R-->>P:
        P-->>U:
    end
    opt Writing to tensor elements
        U-->>P: set_tensor
        P-->>R: 
        R-->>I;
    end
```
