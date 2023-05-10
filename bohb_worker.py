"""
Worker for Examples 1-4
=======================

This class implements a very simple worker used in the firt examples.
"""

import os
import torch
import numpy as np

import ConfigSpace as CS
from hpbandster.core.worker import Worker
from fedtrain_simple import build_model, load_dataset
from server import Server
from tensorboardX import SummaryWriter

class FLWorker(Worker):

    def __init__(self, parent_dir, dataset, batch, eps, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval
        self.parent_dir = parent_dir
        self.dataset = dataset
        self.test_batch, self.tasks_train, self.tasks_test = load_dataset(dataset)    
        self.batch = batch
        self.eps = eps

    def compute(self, config, budget, **kwargs):
        dataset = self.dataset
        parent_dir = self.parent_dir
        test_batch = self.test_batch
        tasks_train = self.tasks_train
        tasks_test = self.tasks_test
        bracket, budget_index, running_index = kwargs['config_id']
        
        for k, v in config.items():
            if '_lr' in k:
                config[k] = 10**v
        # config['server_agg'] = 'weighted' if self.eps == -1 else 'unif'
        config['server_agg'] = 'unif'
        
        server_config = {k.split('_')[1]: v for k, v in config.items() if k.startswith('server')}
        server_config_str = "-".join(f"{k}_{v}" for k, v in server_config.items())
        client_config = {k.split('_')[1]: v for k, v in config.items() if k.startswith('client')}
        client_config_str = "-".join(f"{k}_{v:.4g}" for k, v in client_config.items())
        run_str = f"{dataset}-SERVER-{server_config_str}-CLIENT-{client_config_str}"

        os.makedirs(f"{parent_dir}/{run_str}", exist_ok=True)

        # Load
        last_round = 0
        for d in os.listdir(f"{parent_dir}/{run_str}"):
            if "chkpt" in d:
                last_round = max(last_round, int(d[-6:-3]))
        
        model = build_model(dataset)
        opt_state = None
        if last_round > 0:
            model = torch.load(f"{parent_dir}/{run_str}/chkpt_{last_round:03}.pt")
            opt_state = torch.load(f"{parent_dir}/{run_str}/opt_{last_round:03}.pt")
        
        if last_round > budget:
            model = torch.load(f"{parent_dir}/{run_str}/chkpt_{budget:03}.pt")
            opt_state = torch.load(f"{parent_dir}/{run_str}/opt_{budget:03}.pt")
        
        if last_round > 0:
            os.remove(f"{parent_dir}/{run_str}/chkpt_{last_round:03}.pt")
            os.remove(f"{parent_dir}/{run_str}/opt_{last_round:03}.pt")
        
        model = model.cuda()
        server = Server(model, test_batch, server_config, client_config, opt_state=opt_state)
        writer = SummaryWriter(logdir=f"{parent_dir}/{run_str}")
        # Train
        for epoch in range(last_round, int(budget)):
            train_results, _ = server.communication_round(tasks_train)
            for key in train_results.keys():
                if key.startswith("num"):
                    continue 
                else:
                    total_weight = train_results[f"num_train"].sum()
                    agg_value = train_results[key].sum() / total_weight
                writer.add_scalar(f"Train/{key}", agg_value, epoch)
            
        eval_results, wrongs = server.communication_round(tasks_test, adapt=False, full_eval=True, is_training=False)
        total_weight = eval_results[f"num_train"].sum()
        for key in eval_results.keys():
            if key.startswith("num"):
                continue
            else:
                agg_value = eval_results[key].sum() / total_weight
                unif_agg_value = (eval_results[key] / eval_results[f"num_train"]).sum() / len(tasks_test)
            writer.add_scalar(f"Eval/{key}", agg_value, epoch)
            writer.add_scalar(f"Eval/Uniform{key}", unif_agg_value, epoch)

        server._model.cpu()
        torch.save(server._model, f"{parent_dir}/{run_str}/chkpt_{epoch:03}.pt")
        torch.save(server._opt.state_dict(),  f"{parent_dir}/{run_str}/opt_{epoch:03}.pt")

        n_configs = [
            [5], # 405
            [8, 2], # 135, 405
            [15, 5, 1], # 45, 135, 405
            [34, 11, 3, 1], # 15, 45, 135, 405
            [81, 27, 9, 3, 1] # 5, 15, 45, 135, 405
        ]
        if budget_index == len(n_configs[bracket]) - 1:
        # last rung: select between all rounds trained to 405 = 10 configs
            N = 10
        else:
            N = n_configs[bracket][budget_index+1]
        nc_idx = np.random.choice(len(tasks_test), self.batch, replace=False)
        noise = np.random.laplace(scale=N*11*2/(self.batch*self.eps)) if self.eps != -1 else 0
        # Weighted eval
        # err = (eval_results['GlobalError'][nc_idx].sum() / eval_results['num_train'][nc_idx].sum()) + noise
        # Uniform eval
        normalized_evals = (eval_results['GlobalError'][nc_idx] / eval_results['num_train'][nc_idx])
        err = (normalized_evals.sum() / self.batch) + noise
        err = np.clip(err, 0.0, 1.0)
        return({
            'loss': float(err),  # this is the a mandatory field to run hyperband
            'info': err  # can be used for any user-defined information - also mandatory
        })
    
    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('server_optimizer', ["adam"]))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('server_agg', [""])) # set inside worker init
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('server_batch', [10]))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('server_lr', lower=-6.0, upper=-1.0))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('server_beta1', lower=0, upper=0.9))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('server_beta2', lower=0, upper=0.999))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('client_lr', lower=-6.0, upper=0.0))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('client_batch', [32, 64, 128]))
        return config_space

