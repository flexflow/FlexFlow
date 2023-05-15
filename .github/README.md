# SpecInfer
![build](https://github.com/flexflow/flexflow/workflows/build/badge.svg?branch=master) ![gpu tests](https://github.com/flexflow/flexflow/workflows/gpu-ci/badge.svg?branch=master) ![multinode gpu tests](https://github.com/flexflow/flexflow/workflows/multinode-test/badge.svg?branch=master) ![docker](https://github.com/flexflow/flexflow/workflows/docker-build/badge.svg?branch=master) ![pip](https://github.com/flexflow/flexflow/workflows/pip-install/badge.svg?branch=master) ![shell-check](https://github.com/flexflow/flexflow/workflows/Shell%20Check/badge.svg?branch=master) ![clang-format](https://github.com/flexflow/flexflow/workflows/clang-format%20Check/badge.svg?branch=master) [![Documentation Status](https://readthedocs.org/projects/flexflow/badge/?version=latest)](https://flexflow.readthedocs.io/en/latest/?badge=latest)

SpecInfer is a distributed multi-GPU system for generative large languge model (LLM) inference. SpecInfer accelerates LLM inference using speculative inference and token tree verification. 



## What is SpecInfer



The high computational and memory requirements of generative large language
models (LLMs) make it challenging to serve them quickly and cheaply. 
SpecInfer is an open-source system that accelerates generative LLM
inference with speculative inference and token tree verification. A key insight
behind SpecInfer is to combine various collectively boost-tuned small language
models to jointly predict the LLM’s outputs; the predictions are organized as a
token tree, whose nodes each represent a candidate token sequence. The correctness
of all candidate token sequences represented by a token tree is verified against the
LLM’s output in parallel using a novel tree-based parallel decoding mechanism.
SpecInfer uses an LLM as a token tree verifier instead of an incremental decoder,
which significantly reduces the end-to-end latency and computational requirement
for serving generative LLMs while provably preserving model quality.

## Contributing
Please let us know if you encounter any bugs or have any suggestions by [submitting an issue](https://github.com/flexflow/flexflow/issues).

We welcome all contributions to SpecInfer from bug fixes to new features and extensions.

## Acknowledgements
This project is initiated by members from CMU, Stanford, and UCSD. We will be continuing developing and supporting SpecInfer and the underlying FlexFlow runtime system.

## License
SpecInfer and FlexFlow uses Apache License 2.0.
