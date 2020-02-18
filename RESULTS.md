# Training Logs

## Feb12_21-54-28_hal01_resnet50_gpux64_b208_cpu20_optO2

All Arguments:
Namespace(arch='resnet50', batch_size=208, channels_last=False, data='/home/shared/imagenet/raw/', distributed=True, epochs=90, evaluate=False, gpu=0, keep_batchnorm_fp32=None, local_rank=0, loss_scale=None, lr=5.2, momentum=0.9, opt_level='O2', pretrained=False, print_freq=10, resume='', start_epoch=0, sync_bn=False, weight_decay=0.0001, workers=20, world_size=64)

Training Time: 
1.00 hrs 7.00 mins 51.31 secs | 4071.31 secs


## Feb12_23-28-54_hal01_resnet50_gpux32_b208_cpu20_optO2

All Arguments:
arch='resnet50', batch_size=208, channels_last=False, data='/home/shared/imagenet/raw/', distributed=True, epochs=90, evaluate=False, gpu=0, keep_batchnorm_fp32=None, local_rank=0, loss_scale=None, lr=2.6, momentum=0.9, opt_level='O2', pretrained=False, print_freq=10, resume='', start_epoch=0, sync_bn=False, weight_decay=0.0001, workers=20, world_size=32

Training Time: 
1.00 hrs 50.00 mins 52.91 secs | 6652.91 secs


## Feb09_09-21-23_hal13_resnet50_gpux16_b208_cpu20_optO2

All Arguments:
arch='resnet50', batch_size=208, channels_last=False, data='/home/shared/imagenet/raw/', distributed=True, epochs=90, evaluate=False, gpu=0, keep_batchnorm_fp32=None, local_rank=0, loss_scale=None, lr=1.3, momentum=0.9, opt_level='O2', pretrained=False, print_freq=10, resume='', start_epoch=0, sync_bn=False, weight_decay=0.0001, workers=20, world_size=16

Training Time: 
3.00 hrs 2.00 mins 49.42 secs | 10969.42 secs

Last val: 
Prec@1 75.631 (76.154) Prec@5 92.578 (92.957)

Resource usage:

![](https://paper-attachments.dropbox.com/s_6AE2DA76A07F8AEA0358C4F9706CD9C69343A0EA95A17BD16480AF086F09B55D_1581276177369_gpux16_train.png)

![](https://paper-attachments.dropbox.com/s_6AE2DA76A07F8AEA0358C4F9706CD9C69343A0EA95A17BD16480AF086F09B55D_1581276180402_gpux16_val.png)



## Feb08_13-47-09_hal11_resnet50_gpux8_b208_cpu20_optO2

All Arguments:
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.100.11" --master_port=8888 imagenet_ddp_apex.py -a resnet50 --b 208 --workers 20 --opt-level O2 /home/shared/imagenet/raw/

Training Time: 
5.00 hrs 48.00 mins 24.01 secs | 20904.01 secs

Last val: 
Prec@1 76.292 Prec@5 93.028 

Resource usage:

![](https://paper-attachments.dropbox.com/s_6AE2DA76A07F8AEA0358C4F9706CD9C69343A0EA95A17BD16480AF086F09B55D_1581214097925_gpux8_train.png)

![](https://paper-attachments.dropbox.com/s_6AE2DA76A07F8AEA0358C4F9706CD9C69343A0EA95A17BD16480AF086F09B55D_1581214101759_gpux8_val.png)



## Feb09_14-22-11_hal13_resnet50_gpux4_b208_cpu20_optO2

All Arguments:
arch='resnet50', batch_size=208, channels_last=False, data='/home/shared/imagenet/raw/', distributed=True, epochs=90, evaluate=False, gpu=0, keep_batchnorm_fp32=None, local_rank=0, loss_scale=None, lr=0.325, momentum=0.9, opt_level='O2', pretrained=False, print_freq=10, resume='', start_epoch=0, sync_bn=False, weight_decay=0.0001, workers=20, world_size=4

Training Time: 
10.00 hrs 58.00 mins 56.00 secs | 39536 secs

Last val: 
Prec@1 76.240 Prec@5 93.070



## Feb09_14-20-42_hal14_resnet50_gpux2_b208_cpu20_optO2

All Arguments:
arch='resnet50', batch_size=208, channels_last=False, data='/home/shared/imagenet/raw/', distributed=True, epochs=81, evaluate=False, gpu=0, keep_batchnorm_fp32=None, local_rank=0, loss_scale=None, lr=0.1625, momentum=0.9, opt_level='O2', pretrained=False, print_freq=10, resume='', start_epoch=0, sync_bn=False, weight_decay=0.0001, workers=20, world_size=2

Training Time: 
18.00 hrs 33.00 mins 10.00 secs | 66790 secs (81 epochs)
20.00 hrs 36.00 mins 51.11 secs | 74211.11 secs (90 epochs)

Last val: (stops at 81th epoch)
Prec@1 76.060 Prec@5 92.950

