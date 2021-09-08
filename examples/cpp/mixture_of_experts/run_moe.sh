#!/bin/bash
#SBATCH --job-name=pagerank
#SBATCH --output=slurm.txt
#SBATCH --time=10:00
#SBATCH --nodes=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6000MB
#SBATCH --nodelist=g0001,g0002
#SBATCH --partition=gpu

srun -n 2 ./moe -ll:cpu 4 -ll:gpu 4 -ll:fsize 15000 -ll:zsize 15000 --nodes 2 -ll:util 1 -b 40 -e 1 --search-budget 1 --export strat-tmp.txt
