# an example of running opt model
## how to run?
1. build the flexflow with FF_BUILD_ALL_INFERENCE_EXAMPLES or FF_BUILD_ALL_EXAMPLES
2. download the weight and token file from aws s3. 
```bash
aws s3 cp s3://catalyst-llama/opt_125m_native.tar.gz FF_HOME/examples/cpp/inference/opt/weights

tar -zxvf opt_125m_native.tar.gz
```
3. run *OPT* with `--weights` `--dataset` `--only-data-parallel`
4. run examples/cpp/inference/opt/opt_baseline.py
5. if get same result, it should be fine
## opt default configuration from huggingface opt-125m
```python
OPTConfig {
  "_remove_final_layer_norm": false,
  "activation_function": "relu",
  "attention_dropout": 0.0,
  "bos_token_id": 2,
  "do_layer_norm_before": true,
  "dropout": 0.1,
  "enable_bias": true,
  "eos_token_id": 2,
  "ffn_dim": 3072,
  "hidden_size": 768,
  "init_std": 0.02,
  "layer_norm_elementwise_affine": true,
  "layerdrop": 0.0,
  "max_position_embeddings": 2048,
  "model_type": "opt",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "transformers_version": "4.27.2",
  "use_cache": true,
  "vocab_size": 50272,
  "word_embed_proj_dim": 768
}
```

