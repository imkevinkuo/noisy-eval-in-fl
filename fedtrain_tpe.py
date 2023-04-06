import argparse
from hyperopt import fmin, hp, tpe
import os
import random
import numpy as np
import torch
from fedtrain_simple import build_model, train_model, load_dataset
import json

server_space = {
    'optimizer': 'adam',
    'lr': 10.0 ** hp.uniform('s_lr', low=-6.0, high=-1.0),
    'beta1': hp.uniform('s_beta1', low=0.0, high=0.9),
    'beta2': hp.uniform('s_beta2', low=0.0, high=0.999),
    'rounds': 405,
    'batch': 10, # num. of training clients
    'agg': 'unif',
}

client_space = {
    'lr': 10.0 ** hp.uniform('lr', low=-6.0, high=0.0),
    'momentum': 0.9,
    'weight_decay': 0.00005,
    'batch': hp.choice('batch', [32,64,128]),
    'epochs': 1,
}

space = [server_space, client_space]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-number', '--gpu', default='0', type=str)
    parser.add_argument('--parent-dir', default='runs_tpe', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--batch', default=10, type=int) # num. of eval clients
    parser.add_argument('--eps', default=-1, type=float)
    return parser.parse_args()

def main():
    args = parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    print('Using GPUs: ', os.environ["CUDA_VISIBLE_DEVICES"])

    run_str = f"tpe_seed-{args.seed}"
    train_run_dir = os.path.join(args.parent_dir, "train", run_str)
    evals_run_dir = os.path.join(args.parent_dir, "eval", run_str)
    os.makedirs(train_run_dir, exist_ok=False)
    os.makedirs(evals_run_dir, exist_ok=False)
    with open(os.path.join(train_run_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    test_batch, tasks_train, tasks_test, tasks_test_sizes = load_dataset(args.dataset)
    n_trials = 16
    global t_idx
    t_idx = 0
    def eval_config(config):
        server_config, client_config = config
        
        global t_idx
        t_idx += 1
        run_trial = f"-trial_{t_idx}"
        train_run_dir = os.path.join(args.parent_dir, "train", run_str + run_trial)
        evals_run_dir = os.path.join(args.parent_dir, "eval", run_str + run_trial)
        os.makedirs(train_run_dir, exist_ok=False)
        os.makedirs(evals_run_dir, exist_ok=False)
        model = build_model(args.dataset)
        eval_freq = server_config['rounds']
        eval_results = train_model(
            model, test_batch, server_config, client_config, 
            tasks_train, tasks_test, tasks_test_sizes,
            train_run_dir, evals_run_dir, eval_freq, args.dataset
        )
        # Use args.batch to generate the noisy eval
        if args.batch == len(tasks_test):
            sub_eval = (eval_results["GlobalError"] / eval_results["num_train"]).sum() / args.batch
        else:        
            rand_idx = np.random.permutation(len(tasks_test))[:args.batch]
            sub_eval = (eval_results["GlobalError"][rand_idx] / eval_results["num_train"][rand_idx]).sum() / args.batch
        dp_noise = np.random.laplace(scale=n_trials/(args.batch*args.eps)) if args.eps != -1 else 0
        err = np.clip(sub_eval + dp_noise, 0.0, 1.0)
        return err

    pkl_path = os.path.join(train_run_dir, "hp_trials.pkl")
    best = fmin(
        fn = eval_config,
        space = space, algo=tpe.suggest, 
        max_evals = n_trials,
        trials_save_file = pkl_path
    )

if __name__ == '__main__':
    main()