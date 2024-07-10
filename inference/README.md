# Inference Examples
This folder contains the code to run inference examples in FlexFlow

To create a sample prompt, call (from the `build` folder):

```bash
mkdir -p ../inference/prompt
echo '["San Francisco is a "]' > ../inference/prompt/test.json
```

To download a model for use in C++, call:
```bash
huggingface-cli login # if needed
python ../inference/utils/download_hf_model.py meta-llama/Llama-2-7b-hf --half-precision-only
```

To run the incremental decoding example in C++, call:

```bash
./inference/incr_decoding/incr_decoding -ll:cpu 4 -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model meta-llama/Llama-2-7b-hf -prompt ../inference/prompt/test.json -tensor-parallelism-degree 4
```

To run the speculative inference example in C++, call:

```bash
./inference/spec_infer/spec_infer -ll:cpu 4 -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-160m -prompt ../inference/prompt/test.json -tensor-parallelism-degree 4
```

To run a PEFT model example in C++, call:

```bash
./inference/peft/peft \
    -ll:gpu 4 -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree 4 \
    -ll:fsize 8192 -ll:zsize 12000 \
    -llm-model JackFram/llama-160m \
    -finetuning-dataset ../inference/prompt/peft_dataset.json \
    -peft-model goliaro/llama-160m-lora \
    -enable-peft \
    --use-full-precision \
    --inference-debugging
```