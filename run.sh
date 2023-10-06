python3 ./inference/utils/download_hf_model.py baichuan-inc/baichuan-7B > baichuan.log 2>&1
code baichuan.log 

# # python3 ./inference/utils/download_hf_model.py tiiuae/falcon-7b > falcon.log 2>&1

# # code falcon.log


# python3 ./inference/utils/download_hf_model.py mosaicml/mpt-7b > mpt.log 2>&1

# code mpt.log

./inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:fsize 14000 -ll:zsize 30000 -ll:util 4 -llm-model baichuan-inc/baichuan-7B -prompt ../inference/prompts/chatgpt.json -tensor-parallelism-degree 4  --fusion > baichuan-c.log 2>&1 

./inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:cpu 4 -ll:fsize 14000 -ll:zsize 30000 -ll:util 4 -llm-model decapoda-research/llama-7b-hf -prompt ../inference/prompts/chatgpt.json -tensor-parallelism-degree 4 --fusion 1

gdb ./inference/incr_decoding/incr_decoding

 run -ll:gpu 4  -ll:fsize 14000 -ll:zsize 30000 -ll:util 4 -llm-model baichuan-inc/baichuan-7B -prompt ../inference/prompts/chatgpt.json -tensor-parallelism-degree 4  --fusion 1 

> baichuan-c.log 2>&1 
