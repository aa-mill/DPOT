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
        
        # resolution is the same for all epochs
        resolution = data["resolution"][0]
        
        # avg peak memory across gpus (convert MB to GB)
        memory_per_timestep = [np.mean(epoch)/1024 for epoch in data['peak_memory_per_gpu']]
        # take max 
        util_per_timestep = [np.max(epoch) for epoch in data['gpu_util_per_gpu']]
        
        # calculate means
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

# process data for all GPU configurations
data_1gpu = process_json_files('1gpus')
data_2gpu = process_json_files('2gpus')
data_4gpu = process_json_files('4gpus')

# unpack data
res_1gpu, time_1gpu, mem_1gpu, util_1gpu, std_time_1gpu, std_mem_1gpu, std_util_1gpu = data_1gpu
res_2gpu, time_2gpu, mem_2gpu, util_2gpu, std_time_2gpu, std_mem_2gpu, std_util_2gpu = data_2gpu
res_4gpu, time_4gpu, mem_4gpu, util_4gpu, std_time_4gpu, std_mem_4gpu, std_util_4gpu = data_4gpu

# create subplots
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
colors = ['#FF8A80', '#40C4FF', '#B388FF']

# plot average epoch time
ax[0].errorbar(res_1gpu, time_1gpu, yerr=std_time_1gpu,
            marker='o', linestyle='-', color=colors[0], capsize=5,
            label='1 GPU')
ax[0].errorbar(res_2gpu, time_2gpu, yerr=std_time_2gpu,
            marker='o', linestyle='-', color=colors[1], capsize=5,
            label='2 GPUs')
ax[0].errorbar(res_4gpu, time_4gpu, yerr=std_time_4gpu,
            marker='o', linestyle='-', color=colors[2], capsize=5,
            label='4 GPUs')
ax[0].set_xlabel('Resolution')
ax[0].set_ylabel('Avg. Epoch Time (s)')
ax[0].grid(True)
ax[0].legend()
ax[0].set_xticks(res_1gpu)  # Assuming all have same resolutions

# plot average peak memory
ax[1].errorbar(res_1gpu, mem_1gpu, yerr=std_mem_1gpu,
            marker='o', linestyle='-', color=colors[0], capsize=5,
            label='1 GPU')
ax[1].errorbar(res_2gpu, mem_2gpu, yerr=std_mem_2gpu,
            marker='o', linestyle='-', color=colors[1], capsize=5,
            label='2 GPUs')
ax[1].errorbar(res_4gpu, mem_4gpu, yerr=std_mem_4gpu,
            marker='o', linestyle='-', color=colors[2], capsize=5,
            label='4 GPUs')
ax[1].set_xlabel('Resolution')
ax[1].set_ylabel('Avg. Peak Memory (GB)')
ax[1].grid(True)
ax[1].legend()
ax[1].set_xticks(res_1gpu)

# plot average gpu utilization
ax[2].errorbar(res_1gpu, util_1gpu, yerr=std_util_1gpu,
            marker='o', linestyle='-', color=colors[0], capsize=5,
            label='1 GPU')
ax[2].errorbar(res_2gpu, util_2gpu, yerr=std_util_2gpu,
            marker='o', linestyle='-', color=colors[1], capsize=5,
            label='2 GPUs')
ax[2].errorbar(res_4gpu, util_4gpu, yerr=std_util_4gpu,
            marker='o', linestyle='-', color=colors[2], capsize=5,
            label='4 GPUs')
ax[2].set_xlabel('Resolution')
ax[2].set_ylabel('Avg. GPU Utilization (%)')
ax[2].grid(True)
ax[2].legend()
ax[2].set_xticks(res_1gpu)

# adjust layout and save
plt.tight_layout()
plt.savefig('comparison_plots.png')