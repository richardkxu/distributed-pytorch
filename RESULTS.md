# Training Logs

# gpux8 (orange) vs gpux16 (blue)
![](https://paper-attachments.dropbox.com/s_6AE2DA76A07F8AEA0358C4F9706CD9C69343A0EA95A17BD16480AF086F09B55D_1581276606895_image.png)

![](https://paper-attachments.dropbox.com/s_6AE2DA76A07F8AEA0358C4F9706CD9C69343A0EA95A17BD16480AF086F09B55D_1581276660779_image.png)




# Feb09_09-21-23_hal13_resnet50_gpux16_b208_cpu20_optO2

All Arguments:
arch='resnet50', batch_size=208, channels_last=False, data='/home/shared/imagenet/raw/', distributed=True, epochs=90, evaluate=False, gpu=0, keep_batchnorm_fp32=None, local_rank=0, loss_scale=None, lr=1.3, momentum=0.9, opt_level='O2', pretrained=False, print_freq=10, resume='', start_epoch=0, sync_bn=False, weight_decay=0.0001, workers=20, world_size=16

Training Time: 
3.00 hrs 2.00 mins 49.42 secs | 10969.42 secs

Last val: 
Prec@1 75.631 (76.154) Prec@5 92.578 (92.957)

Resource usage:

![](https://paper-attachments.dropbox.com/s_6AE2DA76A07F8AEA0358C4F9706CD9C69343A0EA95A17BD16480AF086F09B55D_1581276177369_gpux16_train.png)

![](https://paper-attachments.dropbox.com/s_6AE2DA76A07F8AEA0358C4F9706CD9C69343A0EA95A17BD16480AF086F09B55D_1581276180402_gpux16_val.png)





# Feb08_13-47-09_hal11_resnet50_gpux8_b208_cpu20_optO2

All Arguments:
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.100.11" --master_port=8888 imagenet_ddp_apex.py -a resnet50 --b 208 --workers 20 --opt-level O2 /home/shared/imagenet/raw/

Training Time: 
5.00 hrs 48.00 mins 24.01 secs

Last val: 
Prec@1 76.292 Prec@5 93.028 

Resource usage:

![](https://paper-attachments.dropbox.com/s_6AE2DA76A07F8AEA0358C4F9706CD9C69343A0EA95A17BD16480AF086F09B55D_1581214097925_gpux8_train.png)

![](https://paper-attachments.dropbox.com/s_6AE2DA76A07F8AEA0358C4F9706CD9C69343A0EA95A17BD16480AF086F09B55D_1581214101759_gpux8_val.png)





