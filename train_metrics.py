import sys
import os
sys.path.append(['.','./../'])
os.environ ['OMP_NUM_THREADS'] = '16'
import json
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
from accelerate import Accelerator
from timeit import default_timer
from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer import Adam, Lamb
from utils.utilities import count_parameters, load_model_from_checkpoint
from utils.criterion import SimpleLpLoss
from utils.griddataset import MixedTemporalDataset
from utils.make_master_file import DATASET_DICT
from models.fno import FNO2d
from models.dpot import DPOTNet
from models.dpot_res import CDPOTNet
import pynvml
from torch.cuda import max_memory_allocated, max_memory_reserved
from torch.cuda import reset_peak_memory_stats
from itertools import cycle

def get_args():
    parser = argparse.ArgumentParser(description='Training or pretraining for the same data type')
    parser.add_argument('--model', type=str, default='FNO')
    parser.add_argument('--dataset', type=str, default='ns2d')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--train_paths', nargs='+', type=str, default=['ns2d_pdb_M1_eta1e-1_zeta1e-1'])
    parser.add_argument('--test_paths', nargs='+', type=str, default=['ns2d_pdb_M1_eta1e-1_zeta1e-1'])
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--ntrain_list', nargs='+', type=int, default=[9000])
    parser.add_argument('--data_weights', nargs='+', type=int, default=[1])
    parser.add_argument('--use_writer', action='store_true', default=False)
    parser.add_argument('--res', type=int, default=64)
    parser.add_argument('--noise_scale', type=float, default=0.0)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--act', type=str, default='gelu')
    parser.add_argument('--modes', type=int, default=16)
    parser.add_argument('--use_ln', type=int, default=1)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--n_blocks', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=int, default=1)
    parser.add_argument('--out_layer_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'lamb'])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--lr_method', type=str, default='step')
    parser.add_argument('--grad_clip', type=float, default=10000.0)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--step_gamma', type=float, default=0.5)
    parser.add_argument('--warmup_epochs', type=int, default=50)
    parser.add_argument('--sub', type=int, default=1)
    parser.add_argument('--S', type=int, default=64)
    parser.add_argument('--T_in', type=int, default=10)
    parser.add_argument('--T_ar', type=int, default=1)
    parser.add_argument('--T_bundle', type=int, default=1)
    parser.add_argument('--comment', type=str, default="")
    parser.add_argument('--log_path', type=str, default='')
    args = parser.parse_args()
    return args

class ComputeMetricsTracker:
    def __init__(self, accelerator, writer=None):
        self.metrics = {
            'peak_memory_per_gpu': [],
            'gpu_util_per_gpu': [],
            'epoch_time': [],
            'epoch': [],
            'resolution': [],
            'batch_size': []
        }
        self.writer = writer
        pynvml.nvmlInit()
        
        # Get the GPU indices assigned to this process
        self.handles = []
        
        # Check CUDA_VISIBLE_DEVICES regardless of distributed mode
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible_devices:
            # Convert visible_devices string to list of indices
            gpu_indices = [int(x) for x in visible_devices.split(',')]
            for idx in gpu_indices:
                self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(idx))
            print(f"Process tracking GPUs: {gpu_indices}")
        else:
            # Fallback: get current device if CUDA_VISIBLE_DEVICES is not set
            cuda_device = torch.cuda.current_device()
            self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(cuda_device))
            print(f"Process tracking GPU: {cuda_device}")

    def reset(self):
        reset_peak_memory_stats()
        
    def update(self, epoch, resolution, batch_size, epoch_time):
        # Memory and utilization metrics per GPU
        peak_mems = []
        gpu_utils = []
        
        for handle in self.handles:
            # Get memory info for this GPU
            # Get both NVML and PyTorch memory info
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # torch_mem = torch.cuda.max_memory_allocated()/1024**2  # PyTorch's tracked memory in MB
                nvml_mem = meminfo.used/1024**2  # NVML's memory in MB
                
                # Use the larger of the two measurements
                # peak_mems.append(max(torch_mem, nvml_mem))
                peak_mems.append(nvml_mem)
                
                # Debug printing
                # print(f"GPU Memory - PyTorch: {torch_mem:.0f}MB, NVML: {nvml_mem:.0f}MB")
                
                # Take multiple samples of GPU utilization
                utils = []
                for _ in range(3):
                    gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utils.append(gpu_info.gpu)
                    time.sleep(0.1)
                
                gpu_utils.append(max(utils))
        
        metrics = {
            'peak_memory_per_gpu': peak_mems,
            'gpu_util_per_gpu': gpu_utils,
            'epoch_time': epoch_time,
            'epoch': epoch,
            'resolution': resolution,
            'batch_size': batch_size
        }
        
        # Log to tensorboard if available
        if self.writer:
            for gpu_idx, (mem, util) in enumerate(zip(peak_mems, gpu_utils)):
                self.writer.add_scalar(f'compute_metrics/peak_memory_gpu_{gpu_idx}', mem, epoch)
                self.writer.add_scalar(f'compute_metrics/gpu_util_gpu_{gpu_idx}', util, epoch)
            self.writer.add_scalar('compute_metrics/epoch_time', epoch_time, epoch)

        # Store metrics
        for name, value in metrics.items():
            self.metrics[name].append(value)
            
if __name__ == "__main__":
    # get args
    args = get_args()

    # get accelerators
    accelerator = Accelerator(split_batches=False)
    device = accelerator.device

    # set paths
    train_paths = args.train_paths
    test_paths = args.test_paths
    args.data_weights = [1]*len(args.train_paths) if len(args.data_weights) == 1 else args.data_weights

    # get datasets and dataloaders
    train_dataset1 = MixedTemporalDataset([args.train_paths[0]], [args.ntrain_list[0]], res=args.res, 
                                         t_in=args.T_in, t_ar=args.T_ar, normalize=False, 
                                         train=True, data_weights=[args.data_weights[0]])

    train_dataset2 = MixedTemporalDataset([args.train_paths[1]], [args.ntrain_list[1]], res=args.res, 
                                         t_in=args.T_in, t_ar=args.T_ar, normalize=False, 
                                         train=True, data_weights=[args.data_weights[1]])

    test_datasets = [MixedTemporalDataset(test_path, res=args.res, n_channels=train_dataset1.n_channels, t_in=args.T_in, t_ar=-1, normalize=False, train=False) for test_path in test_paths]
    train_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=args.batch_size, 
                                              shuffle=True, num_workers=12, persistent_workers=True)
    train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=args.batch_size, 
                                              shuffle=True, num_workers=12, persistent_workers=True)
    test_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=12) for test_dataset in test_datasets]
    ntrain, ntests = len(train_dataset1), [len(test_dataset) for test_dataset in test_datasets]
    print('Train num {} test num {}'.format(train_dataset1.n_sizes, ntests))
    
    # get model, optimizer, and scheduler
    model = DPOTNet(img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset1.n_channels, in_timesteps = args.T_in, out_timesteps = args.T_bundle, out_channels=train_dataset1.n_channels, normalize=args.normalize, embed_dim=args.width, depth=args.n_layers, n_blocks = args.n_blocks, mlp_ratio=args.mlp_ratio, out_layer_dim=args.out_layer_dim, act=args.act, n_cls=len(args.train_paths)).to(device)
    model = accelerator.prepare(model)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-6)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, div_factor=1e4, pct_start=(args.warmup_epochs / args.epochs), final_div_factor=1e4, steps_per_epoch=len(train_loader1), epochs=args.epochs)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    # set up logging
    comment = args.comment + '_{}_{}'.format(len(train_paths), ntrain)
    log_path = './logs_pretrain/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs_pretrain',args.log_path + comment)
    model_path_fun = lambda epoch: (log_path + '/model_{}.pth'.format(epoch))
    ckpt_save_epochs = 50
    if args.use_writer:
        writer = SummaryWriter(log_dir=log_path)
        fp = open(log_path + '/logs.txt', 'w+',buffering=1)
        json.dump(vars(args), open(log_path + '/params.json', 'w'),indent=4)
        sys.stdout = fp
    else:
        writer = None
    print(model)
    count_parameters(model)

    # prepare training
    train_loader1, train_loader2, *test_loaders = accelerator.prepare(
        train_loader1, train_loader2, *test_loaders
    )

    # Initialize tracker after setting up writer
    tracker = ComputeMetricsTracker(accelerator, writer=writer if args.use_writer else None)
    
    # training loop
    myloss = SimpleLpLoss(size_average=False)
    clsloss = torch.nn.CrossEntropyLoss(reduction='sum')
    # iter = 0
    for ep in range(args.epochs):
        model.train()
        
        t_1 = default_timer()
        t_load, t_train = 0., 0.
        train_l2_step = 0
        train_l2_full = 0
        cls_total, cls_correct, cls_acc = 0, 0, 0.
        loss_previous = np.inf
        samples_processed = 0
        
        tracker.reset()  # Reset peak memory stats at start of epoch
        epoch_start = time.time()

        torch.cuda.empty_cache()

        # use zip to iterate over both dataloaders simultaneously
        n_iters = min(len(train_loader1), len(train_loader2))
        total_batches = np.arange(len(train_loader1) + len(train_loader2))
        for _, batch1, batch2 in zip(total_batches, cycle(train_loader1), cycle(train_loader2)):
            # randomly choose which batch to use
            if torch.rand(1).item() < 0.5:
                xx, yy, msk, cls = batch1
            else:
                xx, yy, msk, cls = batch2
                
            batch_start = time.time()
            t_load += default_timer() - t_1
            t_1 = default_timer()

            loss, cls_loss = 0., 0.
            xx = xx.to(device)
            yy = yy.to(device)
            msk = msk.to(device)
            cls = cls.to(device)

            # auto-regressive training loop
            for t in range(0, yy.shape[-2], args.T_bundle):
                y = yy[..., t:t + args.T_bundle, :]
                xx = xx + args.noise_scale * torch.sum(xx**2, dim=(1,2,3),keepdim=True)**0.5 * torch.randn_like(xx)
                im, cls_pred = model(xx)
                loss += myloss(im, y, mask=msk)

                pred_labels = torch.argmax(cls_pred,dim=1)
                cls_loss += clsloss(cls_pred, cls.squeeze())
                cls_correct += (pred_labels == cls.squeeze()).sum().item()
                cls_total += cls.shape[0]

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=-2)
                xx = torch.cat((xx[..., args.T_bundle:, :], im), dim=-2)

            optimizer.zero_grad()
            total_loss = loss + 0.0 * cls_loss
            accelerator.backward(total_loss)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            # Update metrics
            samples_processed += xx.shape[0] * accelerator.num_processes
            batch_time = time.time() - batch_start
            
            t_train += default_timer() - t_1
            t_1 = default_timer()

        # End of epoch metrics
        epoch_time = time.time() - epoch_start
        avg_samples_per_sec = samples_processed / epoch_time
        
        # Update compute metrics
        tracker.update(
            epoch=ep,
            resolution=args.res,
            batch_size=args.batch_size,
            epoch_time=epoch_time
        )

        # Save metrics every epoch
        metrics_file = os.path.join(log_path, 'compute_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(tracker.metrics, f)
            
        # Save model checkpoint periodically (unchanged)
        if args.use_writer and ep % ckpt_save_epochs == 0:
            torch.save({
                'args': args, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'compute_metrics': tracker.metrics
            }, model_path_fun(ep // ckpt_save_epochs))