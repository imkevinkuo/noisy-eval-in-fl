
import argparse
import os
import pdb
import pickle
import random
from copy import deepcopy
from glob import glob
from heapq import nsmallest
from itertools import product
from math import ceil
from math import log
from operator import itemgetter
import numpy as np
import torch
from numpy.linalg import norm
from scipy.special import logsumexp
from tensorboardX import SummaryWriter
from torch import optim
from yaml import YAMLError

def train(model, X, Y, test_batch, batch=32, dropout=0.0, epochs=1, mu=0.0, **kwargs):
    optimizer = torch.optim.SGD(model.parameters(), **kwargs)
    model.dropout.p = dropout
    model.train()
    m = len(Y)
    # model0 = [param.data.clone() for param in model.parameters()]
    # def prox(model, mu):
    #     return mu*0.5*sum((param-param0).pow(2).sum()
    #     for param, param0 in zip(model.parameters(), model0))
    
    for e in range(epochs):
        randperm = torch.randperm(m)
        X, Y = X[randperm], Y[randperm]
        # for i in range(0, m, batch):
        for i in range(0, 512, batch):
            _, loss, counts, _ = test_batch(model, X[i:i+batch], Y[i:i+batch], with_grad=True)
            loss /= counts.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model

def test(model, X, Y, test_batch, batch=200):
    model.eval()
    agg = []
    client_counts = []
    client_errs = []
    with torch.no_grad():
        for i in range(0, len(Y), batch):
            wrong, loss, counts, errs = test_batch(model, X[i:i+batch], Y[i:i+batch])
            agg.append((wrong, loss))
            client_counts.append(counts)
            client_errs.append(errs)
    total_wrong, total_loss = np.array(agg).sum(axis=0)
    client_counts = torch.cat(client_counts)
    client_errs = torch.cat(client_errs)
    return total_wrong, total_loss, client_counts, client_errs

class Server:
    '''object for federated training implementing methods required by FedEx'''

    def _set_test_state(self):

        state = (np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
        if self._state is None:
            self._state = state
        else:
            np.random.set_state(self._state[0])
            torch.set_rng_state(self._state[1])
            torch.cuda.set_rng_state(self._state[2])
        return state

    def _reset_state(self, state):

        np.random.set_state(state[0])
        torch.set_rng_state(state[1])
        torch.cuda.set_rng_state(state[2])

    def __init__(self, model, test_batch, server_config, client_config=None, state=None, opt_state=None):
        self._model = model
        self._test_batch = test_batch
        if server_config['optimizer'] == 'sgd':
            self._opt = optim.SGD(
                self._model.parameters(), 
                lr=server_config['lr'],
                momentum=server_config['momentum']
            )
        elif server_config['optimizer'] == 'adam':
            self._opt = optim.Adam(
                self._model.parameters(), 
                lr=server_config['lr'], 
                betas=(server_config['beta1'], server_config['beta2'])
            )
        
        if opt_state is not None:
            # self._model.cuda()
            self._opt.load_state_dict(opt_state)
            print('loaded:')
            for k, v in self._opt.state_dict()['state'].items():
                print(" --", v['exp_avg'].device)
            # self._model.cpu()

        self._sched = optim.lr_scheduler.StepLR(self._opt, 1, gamma=0.9999)
        self._batch = server_config['batch']
        self._state = state
        self._reset_state(self._set_test_state())
        self._client_config  = client_config
        self._trace = {key: [] for key in [
            # Global: aggregated model, Local: fine-tuned model
            # For eval clients, we do not fine-tune
            "GlobalError", "GlobalLoss", "LocalError", "LocalLoss"
        ]}
        self._agg = server_config['agg'].lower()
        assert self._agg in ['weighted', 'unif']

    def communication_round(self, clients, num_clients=None, adapt=True, full_eval=False, is_training=True):
        '''runs one step of local training and model aggregation

        Args:
            get_config: returns kwargs for 'train' as a dict
        Returns:
            np.array objects for global val error, local val error, and val size of each client
        '''

        # self._model.cuda()
        if num_clients is None:
            num_clients = len(clients) if full_eval else self._batch
        results = {key: [None for i in range(num_clients)] for key in [
            'num_train', 'GlobalError', 'GlobalLoss', 'LocalError', 'LocalLoss',
        ]}
        result_objects = {key: [None for i in range(num_clients)] for key in [
            'counts', 'errs'
        ]}
        if is_training:
            assert adapt

        # Communication round
        sampled_idx = np.arange(len(clients)) if full_eval else np.random.choice(len(clients), size=num_clients, replace=False)
        
        for i, client_idx in enumerate(sampled_idx):                
            X, Y = clients[client_idx]()
            if Y is None: # NWP task
                Y = X
            results['GlobalError'][i], results['GlobalLoss'][i], client_counts, client_errs = test(self._model, X, Y, self._test_batch)
            results['num_train'][i] = client_counts.sum()
            # results['GlobalEvalError'][i], results['GlobalEvalLoss'][i], results['num_eval'][i] = self._test(self._model, QX, QY)
            if full_eval:
                result_objects['counts'][i] = client_counts
                result_objects['errs'][i] = client_errs
            
            if adapt:
                model = train(deepcopy(self._model), X, Y, self._test_batch, **self._client_config)
                results['LocalError'][i], results['LocalLoss'][i], _, _ = test(model, X, Y, self._test_batch)
            
            # Aggregate personalized models
            if is_training:
                if i:
                    for agg, param in zip(aggregate.parameters(), model.parameters()):
                        if self._agg == 'weighted':
                            agg.data += results['num_train'][i] * param.data
                        else:
                            agg.data += param.data
                else:
                    if self._agg == 'weighted':
                        for param in model.parameters():
                            param.data *= results['num_train'][i]
                    aggregate = model
        
        keys = list(results.keys())
        for k in keys:
            if None in results[k]:
                del results[k]
            else:
                results[k] = np.array(results[k])
        # Model gradient update
        if is_training:
            self._opt.zero_grad()
            for agg, param in zip(aggregate.parameters(), self._model.parameters()):
                if self._agg == 'weighted':
                    param.grad = param.data - agg / results['num_train'].sum()
                else:
                    param.grad = param.data - agg / num_clients 
            self._opt.step()
            self._opt.zero_grad()
            self._sched.step()

        return results, result_objects
