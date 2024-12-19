import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

def process_json_files(folder_path):
    # initialize lists to store results
    resolutions = []
    avg_epoch_time = []
    avg_peak_memory = []
    avg_gpu_util = []
    std_epoch_time = []
    std_peak_memory = []
    std_gpu_util = []

    # get JSON files from the specified folder
    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        # extract resolution from filename (e.g., '16.json' -> 16)
        resolution = int(os.path.splitext(os.path.basename(file))[0])
        
        # avg peak memory across gpus (convert MB to GB)
        memory_per_timestep = [np.mean(epoch)/1024 for epoch in data['peak_memory_per_gpu']]
        # take max utilization across GPUs
        util_per_timestep = [np.max(epoch) for epoch in data['gpu_util_per_gpu']]
        
        # calculate means (skip first epoch)
        avg_time = sum(data['epoch_time'][1:])/len(data['epoch_time'][1:])
        avg_memory = sum(memory_per_timestep[1:])/len(memory_per_timestep[1:])
        avg_util = sum(util_per_timestep[1:])/len(util_per_timestep[1:])
        # calculate std devs
        std_time = np.std(data['epoch_time'][1:])
        std_memory = np.std(memory_per_timestep[1:])
        std_util = np.std(util_per_timestep[1:])
        
        # append to lists
        resolutions.append(resolution)
        avg_epoch_time.append(avg_time)
        avg_peak_memory.append(avg_memory)
        avg_gpu_util.append(avg_util)
        std_epoch_time.append(std_time)
        std_peak_memory.append(std_memory)
        std_gpu_util.append(std_util)

    # sort all data together by resolution
    sorted_data = sorted(zip(resolutions, avg_epoch_time, avg_peak_memory, avg_gpu_util,
                            std_epoch_time, std_peak_memory, std_gpu_util))
    return zip(*sorted_data)

# process data for all configurations
ddp_1gpu = process_json_files('ddp/1gpus')
ddp_2gpu = process_json_files('ddp/2gpus')
ddp_4gpu = process_json_files('ddp/4gpus')
fsdp_1gpu = process_json_files('fsdp/1gpus')
fsdp_2gpu = process_json_files('fsdp/2gpus')
fsdp_4gpu = process_json_files('fsdp/4gpus')

# unpack data
res_ddp1, time_ddp1, mem_ddp1, util_ddp1, std_time_ddp1, std_mem_ddp1, std_util_ddp1 = ddp_1gpu
res_ddp2, time_ddp2, mem_ddp2, util_ddp2, std_time_ddp2, std_mem_ddp2, std_util_ddp2 = ddp_2gpu
res_ddp4, time_ddp4, mem_ddp4, util_ddp4, std_time_ddp4, std_mem_ddp4, std_util_ddp4 = ddp_4gpu
res_fsdp1, time_fsdp1, mem_fsdp1, util_fsdp1, std_time_fsdp1, std_mem_fsdp1, std_util_fsdp1 = fsdp_1gpu
res_fsdp2, time_fsdp2, mem_fsdp2, util_fsdp2, std_time_fsdp2, std_mem_fsdp2, std_util_fsdp2 = fsdp_2gpu
res_fsdp4, time_fsdp4, mem_fsdp4, util_fsdp4, std_time_fsdp4, std_mem_fsdp4, std_util_fsdp4 = fsdp_4gpu

# create two figures
plt.rcParams.update({'font.size': 22})
plt.rcParams['lines.linewidth'] = 3

# first figure with time and memory plots
fig1 = plt.figure(figsize=(24, 10))
gs1 = fig1.add_gridspec(1, 2)
ax1 = [
    fig1.add_subplot(gs1[0, 0]),  # left
    fig1.add_subplot(gs1[0, 1]),  # right
]

# second figure with just utilization
fig2 = plt.figure(figsize=(12, 8))
gs2 = fig2.add_gridspec(1, 1)
ax2 = [fig2.add_subplot(gs2[0, 0])]

# combine axes for easier iteration
ax = ax1 + ax2

colors = ['#FF1744', '#00B0FF', '#7C4DFF']

# get unique x values (resolutions) for tick locations
x_ticks = sorted(set(res_ddp1 + res_ddp2 + res_ddp4))

# set x ticks for all subplots
for subplot in ax:
    subplot.set_xticks(x_ticks)
    subplot.set_xticklabels(x_ticks)

# plot average epoch time
ax[0].errorbar(res_ddp1, time_ddp1, yerr=std_time_ddp1,
            marker='o', linestyle='-', color=colors[0], capsize=5,
            label='1 GPU')
# ax[0].errorbar(res_fsdp1, time_fsdp1, yerr=std_time_fsdp1,
#             marker='s', linestyle='--', color=colors[0], capsize=5,
#             label='1 GPU (FSDP)')
ax[0].errorbar(res_ddp2, time_ddp2, yerr=std_time_ddp2,
            marker='o', linestyle='-', color=colors[1], capsize=5,
            label='2 GPUs (DDP)')
ax[0].errorbar(res_fsdp2, time_fsdp2, yerr=std_time_fsdp2,
            marker='o', linestyle='--', color=colors[1], capsize=5,
            label='2 GPUs (FSDP)')
ax[0].errorbar(res_ddp4, time_ddp4, yerr=std_time_ddp4,
            marker='o', linestyle='-', color=colors[2], capsize=5,
            label='4 GPUs (DDP)')
ax[0].errorbar(res_fsdp4, time_fsdp4, yerr=std_time_fsdp4,
            marker='o', linestyle='--', color=colors[2], capsize=5,
            label='4 GPUs (FSDP)')
ax[0].set_xlabel('Resolution')
ax[0].set_ylabel('Avg. Epoch Time (s)')
ax[0].grid(True)

# plot average peak memory
ax[1].errorbar(res_ddp1, mem_ddp1, yerr=std_mem_ddp1,
            marker='o', linestyle='-', color=colors[0], capsize=5,
            label='1 GPU')
# ax[1].errorbar(res_fsdp1, mem_fsdp1, yerr=std_mem_fsdp1,
#             marker='o', linestyle='--', color=colors[0], capsize=5,
#             label='1 GPU (FSDP)')
ax[1].errorbar(res_ddp2, mem_ddp2, yerr=std_mem_ddp2,
            marker='o', linestyle='-', color=colors[1], capsize=5,
            label='2 GPUs (DDP)')
ax[1].errorbar(res_fsdp2, mem_fsdp2, yerr=std_mem_fsdp2,
            marker='o', linestyle='--', color=colors[1], capsize=5,
            label='2 GPUs (FSDP)')
ax[1].errorbar(res_ddp4, mem_ddp4, yerr=std_mem_ddp4,
            marker='o', linestyle='-', color=colors[2], capsize=5,
            label='4 GPUs (DDP)')
ax[1].errorbar(res_fsdp4, mem_fsdp4, yerr=std_mem_fsdp4,
            marker='o', linestyle='--', color=colors[2], capsize=5,
            label='4 GPUs (FSDP)')
ax[1].set_xlabel('Resolution')
ax[1].set_ylabel('Avg. Peak Memory (GB)')
ax[1].grid(True)

# plot average gpu utilization
ax[2].errorbar(res_ddp1, util_ddp1, yerr=std_util_ddp1,
            marker='o', linestyle='-', color=colors[0], capsize=5,
            label='1 GPU')
# ax[2].errorbar(res_fsdp1, util_fsdp1, yerr=std_util_fsdp1,
#             marker='s', linestyle='--', color=colors[0], capsize=5,
#             label='1 GPU (FSDP)')
ax[2].errorbar(res_ddp2, util_ddp2, yerr=std_util_ddp2,
            marker='o', linestyle='-', color=colors[1], capsize=5,
            label='2 GPUs (DDP)')
ax[2].errorbar(res_fsdp2, util_fsdp2, yerr=std_util_fsdp2,
            marker='o', linestyle='--', color=colors[1], capsize=5,
            label='2 GPUs (FSDP)')
ax[2].errorbar(res_ddp4, util_ddp4, yerr=std_util_ddp4,
            marker='o', linestyle='-', color=colors[2], capsize=5,
            label='4 GPUs (DDP)')
ax[2].errorbar(res_fsdp4, util_fsdp4, yerr=std_util_fsdp4,
            marker='o', linestyle='--', color=colors[2], capsize=5,
            label='4 GPUs (FSDP)')
ax[2].set_xlabel('Resolution')
ax[2].set_ylabel('Avg. GPU Utilization (%)')
ax[2].grid(True)

# after loading other data, add:
def load_single_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # calculate means (skip first epoch)
    avg_time = sum(data['epoch_time'][1:])/len(data['epoch_time'][1:])
    memory_per_timestep = [np.mean(epoch)/1024 for epoch in data['peak_memory_per_gpu']]
    util_per_timestep = [np.max(epoch) for epoch in data['gpu_util_per_gpu']]
    avg_memory = sum(memory_per_timestep[1:])/len(memory_per_timestep[1:])
    avg_util = sum(util_per_timestep[1:])/len(util_per_timestep[1:])
    
    # calculate std devs
    std_time = np.std(data['epoch_time'][1:])
    std_memory = np.std(memory_per_timestep[1:])
    std_util = np.std(util_per_timestep[1:])
    
    return avg_time, avg_memory, avg_util, std_time, std_memory, std_util

# load NRT data
fsdp_plus_1 = load_single_json('1gpus.json')
fsdp_plus_2 = load_single_json('2gpus.json')
fsdp_plus_4 = load_single_json('4gpus.json')

# add horizontal lines for NRT data
for idx, (data, color) in enumerate([(fsdp_plus_1, colors[0]), 
                                   (fsdp_plus_2, colors[1]), 
                                   (fsdp_plus_4, colors[2])]):
    time, mem, util, time_std, mem_std, util_std = data
    gpu_count = 2**idx
    
    # time plot
    ax[0].axhline(y=time, color=color, linestyle=':', label=f'{gpu_count} GPU (NRT)')
    ax[0].fill_between([min(res_ddp1), max(res_ddp1)], 
                      time - time_std, time + time_std,
                      color=color, alpha=0.2)
    
    # memory plot
    ax[1].axhline(y=mem, color=color, linestyle=':', label=f'{gpu_count} GPU (NRT)')
    ax[1].fill_between([min(res_ddp1), max(res_ddp1)], 
                      mem - mem_std, mem + mem_std,
                      color=color, alpha=0.2)
    
    # utilization plot
    ax[2].axhline(y=util, color=color, linestyle=':', label=f'{gpu_count} GPU (NRT)')
    ax[2].fill_between([min(res_ddp1), max(res_ddp1)], 
                      util - util_std, util + util_std,
                      color=color, alpha=0.2)

# reorder legends to group by GPU count
handles, labels = ax[0].get_legend_handles_labels()
order = [3, 0, 4, 5, 1, 6, 7, 2]  # group by GPU count
ax[0].legend([handles[i] for i in order], 
          [labels[i] for i in order])
ax[2].legend([handles[i] for i in order], 
          [labels[i] for i in order])
        #   bbox_to_anchor=(.9, 0.5),  # position legend to right of plots
        #   loc='center left',
        #   borderaxespad=0)

# adjust layout to make room for legend and save
plt.tight_layout()
# plt.subplots_adjust(right=0.85)  # make room for legend
plt.savefig('ddp_fsdp_comparison.png', bbox_inches='tight', dpi=100)

# save both figures
plt.figure(fig1.number)
plt.savefig('ddp_fsdp_comparison_perf.png', bbox_inches='tight', dpi=100)

plt.figure(fig2.number)
plt.savefig('ddp_fsdp_comparison_util.png', bbox_inches='tight', dpi=400)