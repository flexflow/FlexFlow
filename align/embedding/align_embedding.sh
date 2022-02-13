eval "$(conda shell.bash hook)";
conda activate flexflow;
./python/flexflow_python align/embedding/align_embedding_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 1;
conda activate pytorch;
python align/embedding/align_embedding_torch.py -b -v;
python align/embedding/align_embedding.py -b;
