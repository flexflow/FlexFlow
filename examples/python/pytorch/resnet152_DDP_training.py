import os
import tempfile
import torch,torchvision
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.models as models

import time
import pandas as pd


def setup(rank, world_size):
    print("Running init_process_group...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Finished init_process_group...")

def cleanup():
    dist.destroy_process_group()

def train(gpu, args):
    rank = args.nr * args.gpus + gpu	
    setup(rank, args.world_size)
    transform = transforms.Compose([
                torchvision.transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
    batch_size = args.batchsize
    train_dataset = torchvision.datasets.CIFAR10('./datasets/',transform=transform,download=True)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=args.world_size,rank=rank)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2,sampler=sampler)

    model = models.resnet152()
    torch.cuda.set_device(gpu)
    model.cuda()
    print("GPU initialization")
    dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(gpu)
    for _ in range(10):
        _ = model(dummy_input)
    model = nn.parallel.DistributedDataParallel(model,device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    training_run_data = pd.DataFrame(columns=['epoch','batch','batch_size','gpu_number','time'])
    starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        print("Epoch %d"%epoch)
        sampler.set_epoch(epoch)
        for i, data in enumerate(trainloader, 0):
            starter.record()
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ender.record()
            # print statistics
            if rank==0:
                torch.cuda.synchronize()
                timer = starter.elapsed_time(ender)
                training_run_data=training_run_data.append(
                        {'epoch':epoch, 'batch':i,'loss':loss.item(),'batch_size':batch_size,'gpu_number':args.gpus*args.nodes,'time (ms)':timer/(batch_size*args.gpus),'throughput':1000*(batch_size*args.gpus)/timer},
                    ignore_index=True)
                training_run_data.to_csv("training_stats_GPU_%.0f_batchsize_%.0f.csv"%(args.gpus*args.nodes,batch_size),index=False)
                print("[Epoch %d] Batch: %d Loss: %.3f Time per Image: %.2f msi Throughput:%.2f"%
                (epoch,i,loss.item(),timer/(batch_size*args.gpus),1000*(batch_size*args.gpus)/timer))

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
    cleanup()

def main():
    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of epochs')
    parser.add_argument('-b','--batchsize', default=12, type=int) 
    args = parser.parse_args()
    if 'SLURMD_NODENAME' in os.environ:
        if os.environ['SLURMD_NODENAME']==os.environ['MASTER_ADDR']:     
            args.nr=0
        else:
            args.nr=1
    else:
        args.nr=0
    args.world_size = args.gpus * args.nodes   
    print("Spawning processes...")       
    mp.spawn(train, nprocs=args.gpus, args=(args,))

if __name__=='__main__':
    main()
    
