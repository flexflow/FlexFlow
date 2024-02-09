for i in {0..3}; do
    grep "shard:\[$i\]" llama2-cuda-graph-break-down.log | awk -F' ' '{print $3, $4}' > "shard_${i}_ops_detailed.txt"
done