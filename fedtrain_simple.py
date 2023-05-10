import argparse
import json
import os
import pdb
import pickle as pkl
import random
import re
import string
from syslog import LOG_SYSLOG
import numpy as np
import torch; #torch.backends.cudnn.deterministic = False; 
torch.backends.cudnn.benchmark = True
from server import Server
from tensorboardX import SummaryWriter
import models
import itertools
import unicodedata
from transformers import AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

DATA = "./data"

def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--device-number', type=str, default='0',
        help='gpu device number')
    parser.add_argument('--parent-dir', default='runs')
    parser.add_argument('--seed', default=0, type=int, 
        help='Random seed (model init, client sampling)')
    
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--eval-freq', default=15, type=int)

    parser.add_argument('--server-rounds', default=405, type=int)
    parser.add_argument('--server-batch', '--clients-per-round', '--sb', default=10, type=int)
    parser.add_argument('--server-optimizer', '--sopt', default='adam', type=str)
    # Method for aggregating the client parameters: either 'weighted' or 'unif' average
    parser.add_argument('--server-agg', '--sagg', default='weighted', type=str)
    parser.add_argument('--server-lr', '--slr', default=1e-3, type=float)
    parser.add_argument('--server-momentum', '--sm', default=0.9, type=float)
    parser.add_argument('--server-beta1', '--sb1', default=0.9, type=float)
    parser.add_argument('--server-beta2', '--sb2', default=0.999, type=float)

    parser.add_argument('--client-epochs', '--ce', default=1, type=int)
    parser.add_argument('--client-lr', '--clr', default=0.01, type=float)
    parser.add_argument('--client-batch', '--cb', default=64, type=int)
    parser.add_argument('--client-momentum', '--cm', default=0.9, type=float)

    return parser.parse_args()

MAX_SEQ_LEN = 25
max_seq_len = MAX_SEQ_LEN
all_letters = {c: i for i, c in enumerate(string.printable)}
num_letters = len(all_letters)
UNK = num_letters

def unicodeToAscii(self, s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in self.all_letters
    )

def line_to_indices(line: str, max_seq_len: int):
    line_list = split_line(line)  # split phrase in words
    line_list = line_list
    chars = flatten_list([list(word) for word in line_list])
    indices = [
        all_letters.get(letter, UNK)
        for i, letter in enumerate(chars)
        if i < max_seq_len
    ]
    # Add padding
    indices = indices + [UNK] * (max_seq_len - len(indices))
    return indices

def process_x(raw_x_batch):
    x_batch = [e[4] for e in raw_x_batch]  # e[4] contains the actual tweet
    x_batch = [line_to_indices(e, max_seq_len) for e in x_batch]
    return x_batch

def process_y(raw_y_batch):
    y_batch = [int(e) for e in raw_y_batch]
    return y_batch

def split_line(line):
    """
    Split given line/phrase (str) into list of words (List[str])
    """
    return re.findall(r"[\w']+|[.,!?;]", line)

def flatten_list(nested_list):
    return list(itertools.chain.from_iterable(nested_list))

def get_loader(task):
    dump = torch.load(task)
    X, Y = dump['X'].pin_memory(), dump['Y'].pin_memory()
    def loader():
        return X.cuda(non_blocking=True), Y.cuda(non_blocking=True)
    return loader, len(X)


def load_tasks(json_path):
    tasks = []
    task_sizes = []
    dataset = json_path.split('/')[1]
    with open(json_path) as f:
        data = json.load(f)
    for i, (client_id, client_size) in enumerate(data.items()):
        print('\rCaching task', i+1, end='')
        task = f"{DATA}/{dataset}/cache/{client_id}.pt"
        def get_loader(task):
            dump = torch.load(task)
            X = dump['X'].pin_memory()
            Y = dump['Y'].pin_memory() if 'Y' in dump else None
            def loader():
                return X.cuda(non_blocking=True), None if Y is None else Y.cuda(non_blocking=True)
            return loader
        
        if os.path.isfile(task):
            loader = get_loader(task)
        else:
            raise Exception(f"{task} does not exist")
        
        tasks.append(loader)
        task_sizes.append(client_size)

    print("Loaded", len(tasks), "tasks")
    
    return tasks, task_sizes

def test_batch_image(model, X, Y, with_grad=False):
    """ 
    evals a single client
    wrong: total num. of wrong examples
    loss: total loss over all examples
    counts: tensor of ones
    errs: 0/1 tensor indicating which examples are predicted wrong
    """
    pred = model(X)
    errs = Y != pred.argmax(1)
    wrong = errs.sum().float().cpu().numpy()
    loss = torch.nn.CrossEntropyLoss(reduction='sum')(pred, Y).float()
    counts = torch.ones(len(Y)) # weight of each example - all 1 for image datasets
    if not with_grad:
        loss = loss.detach().cpu().numpy()
    return wrong, loss, counts, errs.cpu()

def test_batch_text(model, X, Y, with_grad=False, with_err=True):
    """ 
    evals a single client
    wrong: total num. of wrong tokens
    loss: total loss over all tokens
    counts: num. of total tokens
    errs: num. of wrong tokens in each example
    """

    with torch.set_grad_enabled(with_grad):
        output = model(X)
        if isinstance(model, GPT2LMHeadModel): # DistilGPT2
            logits = output.logits[:, :-1]
        else: # LSTM
            logits = output[:, :-1]
        flat_logits = logits.reshape(-1, 50257) # exclude last token
        loss = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(flat_logits, dim=-1), # flat predictions
            Y[:, 1:].reshape(-1), # flat tokens
            # comment out for huggingface loss = model(X, labels=X).loss
            ignore_index=50256,
            reduction='sum'
        )
    non_pad_idx = Y[:, 1:] != 50256                           # [B, S]: bool
    counts = non_pad_idx.sum(dim=1)                           # [sentences]: int
    if with_err:
        with torch.no_grad():
            pred_toks = logits.argmax(dim=-1)                 # [sentences, tokens]: 0...50256
            wrong_toks = pred_toks != Y[:, 1:]                # [sentences, tokens]: bool
            errs = (non_pad_idx*wrong_toks).sum(dim=1)        # [sentences]: int
            wrong = errs.sum().cpu().numpy()
    else:
        wrong = None
        errs = None
    
    if not with_grad:
        loss = loss.cpu().numpy()

    return wrong, loss, counts.cpu(), errs.cpu()

def train_model(model, test_batch, server_config, client_config, 
                tasks_train, tasks_test,
                logdir_train, logdir_evals, eval_freq, dataset=None):
    model = model.cuda()
    writer = SummaryWriter(logdir_train)
    server = Server(model, test_batch, server_config, client_config)
    rounds = server_config['rounds']
    for epoch in range(rounds + 1):
        # Run Train
        train_results, _ = server.communication_round(tasks_train)
        for key in train_results.keys():
            if key.startswith("num"):
                continue 
            else:
                total_weight = train_results[f"num_train"].sum()
                agg_value = train_results[key].sum() / total_weight
            writer.add_scalar(f"Train/{key}", agg_value, epoch)
            server._trace[key].append(agg_value)
        
        if epoch % eval_freq == 0: # default: 15, 405 total rounds
            pass
        elif epoch == 5: # For Hyperband
            pass
        else:
            continue

        # save_rounds = []
        # if epoch in save_rounds:
        #     server._model.cpu()
        #     torch.save(server._model, f"{logdir_train}/chkpt_{epoch:03}.pt")
        #     server._model.cuda()
        
        # Run Eval Clients
        eval_results, result_objects = server.communication_round(tasks_test, adapt=False, full_eval=True, is_training=False)
        total_weight = eval_results[f"num_train"].sum()
        for key in eval_results.keys():
            if key.startswith("num"):
                continue
            else:
                agg_value = eval_results[key].sum() / total_weight
                unif_agg_value = (eval_results[key] / eval_results[f"num_train"]).sum() / len(tasks_test)
            writer.add_scalar(f"Eval/{key}", agg_value, epoch)
            writer.add_scalar(f"Eval/Uniform{key}", unif_agg_value, epoch)
        
        filename = f"{logdir_evals}/R{epoch}.pkl"
        with open(filename, 'wb') as f:
            pkl.dump(eval_results, f)
        
        # shuffle
        client_sizes = [len(client_counts) for client_counts in result_objects['counts']]
        assert len(client_sizes) == len(tasks_test)
        flat_counts = torch.cat(result_objects['counts']) # each client has a vector -> cat into one vector
        flat_errs = torch.cat(result_objects['errs'])
        for p in [0.5, 1]:
            perm_eval_results = {'GlobalError': [], 'num_train': []}
            eval_perm = np.load(f"data/{dataset}/eval_perm_p={p}.npy")
            perm_errs = flat_errs[eval_perm]
            perm_counts = flat_counts[eval_perm]
            curr = 0
            for size in client_sizes:
                perm_eval_results['GlobalError'].append(perm_errs[curr:curr+size].sum())
                perm_eval_results['num_train'].append(perm_counts[curr:curr+size].sum())
                curr += size           
            
            perm_eval_results['GlobalError'] = np.array(perm_eval_results['GlobalError'])
            perm_eval_results['num_train'] = np.array(perm_eval_results['num_train'])
            filename = f"{logdir_evals}/P{p}_R{epoch}.pkl"
            with open(filename, 'wb') as f:
                pkl.dump(perm_eval_results, f)
    
    return eval_results

def load_dataset(dataset):
    if dataset == 'cifar10' or dataset == 'femnist':
        test_batch = test_batch_image
    elif dataset == 'stackoverflow' or dataset == 'reddit':
        test_batch = test_batch_text
    else:
        raise Exception("invalid dataset")
    
    tasks_train, _ = load_tasks(f"data/{dataset}/train_clients.json")
    tasks_test, _ = load_tasks(f"data/{dataset}/eval_clients.json")
    for p in [0.5, 1]:
        eval_perm_pt = f"{DATA}/{dataset}/eval_perm_p={p}.npy"
        if not os.path.exists(eval_perm_pt):
            raise Exception(
                f"File {eval_perm_pt} not found. Generate it using notebooks/generate_data.ipynb or download it from the Github repo."
            )
    return test_batch, tasks_train, tasks_test

def build_model(dataset):
    if "cifar10" in dataset:
        model = models.CNN(in_dim=(3, 32, 32), out_dim=10)
    elif "femnist" in dataset:
        model = models.CNN(in_dim=(1, 28, 28), out_dim=62)
    elif "sent140" in dataset:
        model = models.LSTM(
            num_classes=2,
            n_hidden=100,
            num_embeddings=100 + 1,
            embedding_dim=100,
            max_seq_len=MAX_SEQ_LEN,
            dropout_rate=0.1,
        )
    elif "stackoverflow" in dataset:
        # model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        # for name, p in model.named_parameters():
        #     p.requires_grad = False
        # model.lm_head.weight.requires_grad = True
        model = models.LSTM(
            num_classes=50257,
            n_hidden=128,
            num_embeddings=50257,
            embedding_dim=128,
            max_seq_len=MAX_SEQ_LEN,
            dropout_rate=0.1,
            full_seq=True
        )
    elif "reddit" in dataset:
        model = models.LSTM(
            num_classes=50257,
            n_hidden=128,
            num_embeddings=50257,
            embedding_dim=128,
            max_seq_len=MAX_SEQ_LEN,
            dropout_rate=0.1,
            full_seq=True
        )
    elif "synthetic" in dataset:
        model = models.LogisticRegression(60,5)
    return model

def main():

    args = parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    print('Using GPUs: ', os.environ["CUDA_VISIBLE_DEVICES"])

    # Set up data and model
    test_batch, tasks_train, tasks_test = load_dataset(args.dataset)
    model = build_model(args.dataset)

    # Logging
    server_config = {k.split('_')[1]: v for k, v in vars(args).items() if k.startswith('server')}
    server_config_str = "-".join(f"{k}_{v:.4g}" if isinstance(v, float) else f"{k}_{v}" for k, v in server_config.items())
    client_config = {k.split('_')[1]: v for k, v in vars(args).items() if k.startswith('client')}
    client_config_str = "-".join(f"{k}_{v:.4g}" if isinstance(v, float) else f"{k}_{v}" for k, v in client_config.items())
    run_str = f"{args.dataset}-SEED-{args.seed}-SERVER-{server_config_str}-CLIENT-{client_config_str}"
    
    train_run_dir = os.path.join(args.parent_dir, "train", run_str)
    evals_run_dir = os.path.join(args.parent_dir, "eval", run_str)
    os.makedirs(train_run_dir, exist_ok=False)
    os.makedirs(evals_run_dir, exist_ok=False)
    with open(os.path.join(train_run_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Train
    train_model(
        model, test_batch, server_config, client_config, 
        tasks_train, tasks_test,
        train_run_dir, evals_run_dir, args.eval_freq, args.dataset
    )


if __name__ == '__main__':
    main()
