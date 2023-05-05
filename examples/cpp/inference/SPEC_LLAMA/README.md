# an example of running llama model with beam search

## how to run it?
1. build the flexflow with FF_BUILD_ALL_INFERENCE_EXAMPLES or FF_BUILD_ALL_EXAMPLES
2. download the weight and token file from aws s3. 
```bash
aws s3 cp s3://catalyst-llama/7B_weights_float.tar.gz FF_HOME/examples/cpp/inference/SPEC_LLAMA/weights
tar -zxvf 7B_weights_float.tar.gz 
aws s3 cp s3://catalyst-llama/tokens.tar FF_HOME/examples/cpp/inference/SPEC_LLAMA/tokens
tar -zxvf tokens.tar
```
3. run *SPEC_LLAMA* with `--weights` `--dataset`  `-b 5` `--only-data-parallel`
4. [expected results](https://github.com/flexflow/FlexFlow/pull/681#issuecomment-1534264054)

