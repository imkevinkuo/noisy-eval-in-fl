import os
import subprocess
import time
from sys import stdout

FNULL = open(os.devnull, 'w')

def run_script(args_dict, gpu):

    main_command = [
        'python3',
        'fedtrain_simple.py',
    ]

    args_dict.update({'device-number': str(gpu)})
    

    whole_command = main_command
    for k, v in args_dict.items():
        whole_command.append('--' + k)
        whole_command.append(v)

    print(whole_command)
    subprocess.Popen(args=whole_command, stdout=FNULL)

import subprocess
import numpy as np
def min_gpu(gpus, threshold):
    """
    Runs 'nvidia-smi' and finds how many processes are using each GPU. 
    Returns the GPU with fewest running processes.
    If more than 'threshold' processes are using the GPU, return -1. 
    """
    device_counts = [0 for _ in range(gpus)]
    process_output = subprocess.run(['nvidia-smi'], capture_output=True)
    output_lines = process_output.stdout.split(b"|\n")
    i = -2 # line idx
    curr_line = str(output_lines[i])
    while '=' not in curr_line and "No running processes found" not in curr_line:
        device_idx = int(output_lines[i].split()[1])
        device_counts[device_idx] += 1
        i -= 1
        curr_line = str(output_lines[i])
    argmin = np.argmin(device_counts)
    if device_counts[argmin] >= threshold:
        return -1
    else:
        return argmin

def acquire_gpu(gpus=8, threshold=1):
    i = 0
    gpu = min_gpu(gpus=gpus, threshold=threshold)
    while gpu == -1:
        stdout.write("\rWaiting on gpu for: {0}s".format(i))
        i += 30
        time.sleep(30)
        gpu = min_gpu(gpus=gpus, threshold=threshold)
    stdout.write("\n")
    return gpu

import numpy as np

SERVER = lambda: {
    'server-optimizer': 'adam',
    'server-lr': 10.0 ** np.random.uniform(low=-6.0, high=-1.0), # For search space experiment, test -10 to -6
    'server-beta1': np.random.uniform(low=0.0, high=0.9),
    'server-beta2': np.random.uniform(low=0.0, high=0.999), 
    'server-batch': 10,
}

CLIENT = lambda: {
    'client-lr': 10.0 ** np.random.uniform(low=-6.0, high=0.0), #HP4
    'client-momentum': 0.9,
    'client-batch': np.random.choice([32,64,128]),
    'client-epochs': 1,
}
args_keys = [
    'parent-dir',
    'seed',
    'dataset',
    'eval-freq',
    'server-agg',
]

args_dicts = []

# datasets = ["cifar10", "femnist", "stackoverflow", "reddit"]
datasets = ["cifar10", "femnist"]
for dataset in datasets:
    for agg in ['weighted', 'unif']:
        for seed in range(128):
            np.random.seed(seed)
            server_config = SERVER()
            client_config = CLIENT()
            args_values = (f"runs_{agg}/runs_simple/runs_{dataset}", str(seed), dataset, '15', agg)
            args_dict = dict(zip(args_keys, args_values))
            args_dict.update({k: str(v) for k, v in server_config.items()})
            args_dict.update({k: str(v) for k, v in client_config.items()})
            args_dicts.append(args_dict)

for args_dict in args_dicts:
    gpu = acquire_gpu(threshold=1)
    run_script(args_dict, gpu)
    time.sleep(10)
    # time.sleep(5)

FNULL.close()