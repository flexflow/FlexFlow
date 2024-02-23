    ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 > llama2-cuda-graph-break-down.log 2>&1 

    # -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/serving/inference/prompt/chatgpt.json -tensor-parallelism-degree 4

 ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 > llama2-cuda-graph-bs64-tp4.log 2>&1 
 
    nsys profile  -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o  cuda_graph_node_tp4_bs2_0215 -f true -x true --cuda-graph-trace node  ./build/inference/incr_decoding/incr_decoding  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -ll:gpu 4 -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4


#nsys profile  -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o  cuda_graph_graph -f true -x true --cuda-graph-trace graph  ./build/inference/incr_decoding/incr_decoding  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -ll:gpu 4 -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4


#./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/chatgpt.json  -pipeline-parallelism-degree 4 > 0215-catalyst-llama2-cuda-graph-bs2-pp4.log 2>&1


    ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/full_chatgpt.json -tensor-parallelism-degree 4 > 0223-catalyst-llama2-cuda_graph-bs64-tp4-fullchatgpt.log 2>&1 


    ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 > 0223-catalyst-llama2-cuda_graph-bs64-tp4-prompt2.log 2>&1 


    nsys profile  -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o  cuda_graph_node_tp4_0223_1754 -f true -x true --cuda-graph-trace node  ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 > 0223-catalyst-llama2-cuda_graph-bs64-tp4-prompt2-profile.log 2>&1 


    #./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 > 0223-catalyst-llama2-cuda_graph-bs64-tp4-prompt2-generate256.log 2>&1 


  # nsys profile  -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o  cuda_graph_node_tp4_bs2_0223_generate256 -f true -x true --cuda-graph-trace node   ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 > 0223-catalyst-llama2-cuda_graph-bs64-tp4-prompt2-generate256-profile.log 2>&1 



    ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsie 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/full_chatgpt.json -tensor-parallelism-degree 4 > 0223-1754-llama2-cuda_graph-fulldataset.log 2>&1 


     nsys profile  -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o  0223-1754-cuda-graph-fulldataset -f true -x true --cuda-graph-trace node  ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/full_chatgpt.json -tensor-parallelism-degree 4 > 0223-1754-llama2-cuda_graph-fulldataset-profile.log 2>&1 


    ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 > 0223-1754-llama2-cuda_graph-11prompt-printbc.log 2>&1 


     nsys profile  -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o  0223-1754-cuda-graph-11prompt -f true -x true --cuda-graph-trace node  ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 > 0223-1754-llama2-cuda_graph-11prompt-profile.log 2>&1 



./build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-68m -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 --fusion -cache-folder /home/xiaoxias/weights > 0223-1754-llama2-cuda_graph-spec-infer-11prompt-printbc.log 2>&1 


#set args -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-68m -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 --fusion -cache-folder /home/xiaoxias/weights 


./build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-68m -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 --fusion -cache-folder /home/xiaoxias/weights > 0223-1754-llama2-cuda_graph-spec-infer-11prompt-printbc.log 2>&1 



./build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-68m -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 --fusion -cache-folder /home/xiaoxias/weights > 0223-1754-llama2-cuda_graph-spec-infer-11prompt-zeyu.log 2>&1 


    ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4 > 0223-1754-llama2-cuda_graph-11prompt-zeyu.log 2>&1 

#     ./build/inference/incr_decoding/incr_decoding -ll:gpu 4  -ll:cpu 4 -ll:fsize 20000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -cache-folder /home/xiaoxias/weights --fusion -prompt /home/xiaoxias/chatgpt.json -tensor-parallelism-degree 4