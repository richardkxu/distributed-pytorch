# Distributed training with PyTorch

## Overview

We will cover the following training methods for PyTorch:
- regular, single node, single GPU training
- `torch.nn.DataParallel`
- `torch.nn.DistributedDataParallel`
- mixed precision training

We will cover the following use cases:
- Single node single GPU training
- Single node multi-GPU training
- Multi-node multi-GPU training

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org)) with GPU
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
    - on HAL cluster, use `/home/shared/imagenet/raw/`

## Non-distributed (ND) training

Use cases:
- Single node single GPU training

This is the most basic training method, no data parallel at all

```bash
python imagnet_nd.py -a resnet50 /home/shared/imagenet/raw/
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

```bash
python imagnet_nd.py -a alexnet --lr 0.01 /home/shared/imagenet/raw/
```

## Single-processing Data Parallel (DP)

Use cases:
- Single node multi-GPU training

References:

- https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html

`torch.nn.DataParallel` is easier to use (just wrap the model and run your training script). However, because it uses one process to compute the model weights and then distribute them to each GPU on the current node during each batch, networking quickly becomes a bottle-neck and GPU utilization is often very low. Furthermore, it requires that all the GPUs be on the same node and doesn’t work with `Apex` for mixed-precision training.

## Multi-processing Distributed Data Parallel (DDP)

Use cases:
- Single node multi-GPU training
- Multi-node multi-GPU training

References: 
- https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
- https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py
- https://github.com/pytorch/examples/blob/master/imagenet/main.py
- https://medium.com/intel-student-ambassadors/distributed-training-of-deep-learning-models-with-pytorch-1123fa538848
- http://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/
- http://seba1511.net/dist_blog/

`torch.nn.DistributedDataParallel` is the recommeded way of doing distributed training in PyTorch. It is proven to be significantly faster than `torch.nn.DataParallel` for single-node multi-GPU data parallel training. `nccl` backend is currently the fastest and highly recommended backend to be used with distributed training and this applies to both single-node and multi-node distributed training.

Multiprocessing with DistributedDataParallel duplicates the model on each GPU on each compute node. The GPUs can all be on the same node or spread across multiple nodes. If you have 2 computer nodes with 4 GPUs each, you have a total of 8 model replicas. Each replica is controlled by one process and handles a portion of the input data.  Every process does identical tasks, and each process communicates with all the others. During the backwards pass, gradients from each node are averaged. Only gradients are passed between the processes/GPUs so that network communication is less of a bottleneck.

During training, each process loads its own minibatches from disk and passes them to its GPU. Each GPU does its own forward pass, and then the gradients are all-reduced across the GPUs. Gradients for each layer do not depend on previous layers, so the gradient all-reduce is calculated concurrently with the backwards pass to futher alleviate the networking bottleneck. At the end of the backwards pass, every node has the averaged gradients, ensuring that the model weights stay synchronized.

All this requires that the multiple processes, possibly on multiple nodes, are synchronized and communicate. Pytorch does this through its `distributed.init_process_group` function. This function needs to know where to find process 0 so that all the processes can sync up and the total number of processes to expect. Each individual process also needs to know the total number of processes as well as its rank within the processes and which GPU to use. It’s common to call the total number of processes the `world size`. Finally, each process needs to know which slice of the data to work on so that the batches are non-overlapping. Pytorch provides `nn.utils.data.DistributedSampler` to accomplish this.


### Single node, multiple GPUs:

```bash
python  imagenet_ddp.py -a resnet50 --dist-url 'tcp://MASTER_IP:MASTER_PORT' --dist-backend 'nccl' --world-size 1 --rank 0 --desired-acc 0.75 /home/shared/imagenet/raw/
```

### Multiple nodes, multiple GPUs:

To run your programe on 4 nodes with 4 GPU each, you will need to open 4 terminals and run slightly different command on each node.

Node 0:
```bash
python  imagenet_ddp.py -a resnet50 --dist-url 'tcp://MASTER_IP:MASTER_PORT' --dist-backend 'nccl' --world-size 4 --rank 0 --desired-acc 0.75 /home/shared/imagenet/raw/
```

- `MASTER_IP`: IP address for the master node of your choice
- `MASTER_PORT`: open port number on the master node. if you don't know, use `8888`
- `--world-size`: equals the # of compute nodes you are using
- `--rank`: rank of the current node, should be an int between `0` and `--world-size - 1`
- `--desired-acc`: desired accuracy to stop training

Node 1:
```bash
python  imagenet_ddp.py -a resnet50 --dist-url 'tcp://MASTER_IP:MASTER_PORT' --dist-backend 'nccl' --world-size 4 --rank 1 --desired-acc 0.75 /home/shared/imagenet/raw/
```

Node 2:
```bash
python  imagenet_ddp.py -a resnet50 --dist-url 'tcp://MASTER_IP:MASTER_PORT' --dist-backend 'nccl' --world-size 4 --rank 2 --desired-acc 0.75 /home/shared/imagenet/raw/
```

Node 3:
```bash
python  imagenet_ddp.py -a resnet50 --dist-url 'tcp://MASTER_IP:MASTER_PORT' --dist-backend 'nccl' --world-size 4 --rank 3 --desired-acc 0.75 /home/shared/imagenet/raw/
```

## FP16 and FP32 mixed precision training with NVIDIA `Apex`

References:

- mnist apex: https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-mixed.py
- apex examples: https://github.com/nvidia/apex/tree/master/examples
- apex tutorial: https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/
- apex tutorial: https://developer.nvidia.com/automatic-mixed-precision
- apex doc: https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html

Mixed precision training: majority of the network uses FP16 arithmetic, while automatically casting potentially unstable operations to FP32.

Key points:
- Ensuring that weight updates are carried out in FP32.
- Loss scaling to prevent underflowing gradients.
- A few operations (e.g. large reductions) left in FP32.
- Everything else (the majority of the network) executed in FP16.

Advantages:
- reducing memory storage/bandwidth demands by 2x
- use larger batch sizes
- take advantage of NVIDIA Tensor Cores for matrix multiplications and convolutions
- don't need to explicitly convert your model, or the input data, to half().

Use `imagenet_ddp_mixprec.py` for training. It is run the same way as the DDP training script.


## Distributed training with `Horovod`

References:

- https://docs.databricks.com/applications/deep-learning/distributed-training/mnist-pytorch.html
- https://horovod.readthedocs.io/en/latest/pytorch.html
- https://github.com/horovod/horovod/blob/master/docs/pytorch.rst

## Acknowledgment

Shout-out to all the references, blogs, code samples, figures... used in this tutorial!
