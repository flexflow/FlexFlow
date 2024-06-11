## Peft Demo
* `git clone -b peft --recursive https://github.com/flexflow/FlexFlow.git`
* `cd FlexFlow/`

* If you wish to run the demo by installing FlexFlow
    * `conda env create -f conda/flexflow.yml`
    * `conda activate flexflow`

* If you wish to run the demo using a Docker container
    * `export FF_CUDA_ARCH=all && export cuda_version=12.0 && ./docker/build.sh flexflow && ./docker/run.sh flexflow`

* Then, install the Llama2 model (the `meta-llama/Llama-2-7b-hf` model is gated, so make sure to add your HF access token)

    * `export HUGGINGFACE_TOKEN="[Your token]"`
    * `huggingface-cli login --token "$HUGGINGFACE_TOKEN"`
    * `python3 inference/utils/download_peft_model.py "goliaro/llama-2-7b-lora-full" --base_model_name "meta-llama/Llama-2-7b-hf"`

* Run the demo
    ```
    mkdir inference/output
    cd inference/python/peft_demo/
    python3 demo.py -config-file demo_config.json
    ```


