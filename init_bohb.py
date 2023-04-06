import os
import subprocess
import time
from sys import stdout

trials = 8
dataset = "cifar10" # cifar10, femnist, stackoverflow, reddit

FNULL = open(os.devnull, 'w')

def run_script(args_dict, gpu, blocking=False):

    main_command = [
        'python3',
        'fedtrain_bohb.py',
    ]

    args_dict.update({'device-number': str(gpu)})
    

    whole_command = main_command
    for k, v in args_dict.items():
        whole_command.append('--' + k)
        whole_command.append(v)

    print(whole_command)
    if blocking:
        subprocess.call(args=whole_command, stdout=FNULL)
    else:
        subprocess.Popen(args=whole_command, stdout=FNULL)

import numpy as np

args_keys = [
    'parent-dir',
    'seed',
    'dataset',
    'batch',
    'eps',
]

args_dicts = []

c_dict = {
    'cifar10': (1, 100),
    'femnist': (3, 360),
    'stackoverflow': (36, 3678),
    'reddit': (100, 10000)
}

# fedtrain_bohb.py uses uniform training objective
sub_c, max_c = c_dict[dataset]
for seed in range(8):
    np.random.seed(seed)
    args_values = (f"runs_unif/runs_bohb/runs_{dataset}/seed_{seed}", str(seed), dataset, str(max_c), '-1')
    # args_values = (f"runs_unif/runs_bohb/runs_{dataset}_s_e=1/seed_{seed}", str(seed), dataset, str(sub_c), '1')
    args_dict = dict(zip(args_keys, args_values))
    args_dicts.append(args_dict)

    # For a machine with multiple GPUs
    # gpu = seed
    # run_script(args_dict, gpu)
    # For a machine with a single GPU
    gpu = 0
    run_script(args_dict, gpu, blocking=True)
    time.sleep(5)


FNULL.close()