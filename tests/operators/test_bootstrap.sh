#!/bin/bash
# request 8 gpus resource for testing
srun --nodes=1 --gres=gpu:8 --cpus-per-task=80 --partition=dev --time=30 --pty /bin/bash -l