import logging
logging.basicConfig(level=logging.WARNING)

import argparse
import torch
import numpy as np
import random
import os

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker
from bohb_worker import FLWorker

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device-number', type=str, default='0',
        help='gpu device number')
    parser.add_argument('--parent-dir', default='bohb_seed_0')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=5)
    parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=405)
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=5)
    #
    parser.add_argument('--batch', default=1, type=int)
    parser.add_argument('--eps', default=-1, type=float)
    return parser.parse_args()
def main():

    args = parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    print('Using GPUs: ', os.environ["CUDA_VISIBLE_DEVICES"])
    os.makedirs(args.parent_dir)

    host = f"127.0.0.1"
    port = 6014+args.seed
    NS = hpns.NameServer(run_id='example1', host=host, port=port)
    NS.start()
    w = FLWorker(
        sleep_interval = 0,
        nameserver=host,
        nameserver_port=port,
        run_id='example1',
        parent_dir=args.parent_dir,
        dataset=args.dataset,
        batch=args.batch,
        eps=args.eps,
    )
    w.run(background=True)
    bohb = BOHB(  configspace = w.get_configspace(),
                run_id='example1', nameserver=host, nameserver_port=port,
                min_budget=args.min_budget, max_budget=args.max_budget
            )
    res = bohb.run(n_iterations=args.n_iterations)

    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    import pickle as pkl
    with open(f"{args.parent_dir}/bohb_results.pkl", 'wb') as f:
        pkl.dump(res, f)

    # Step 5: Analysis
    # Each optimizer returns a hpbandster.core.result.Result object.
    # It holds informations about the optimization run like the incumbent (=best) configuration.
    # For further details about the Result object, see its documentation.
    # Here we simply print out the best config and some statistics about the performed runs.
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    # cleanup
    for run in os.listdir(args.parent_dir):
        if run.endswith(".pkl"):
            continue
        for fn in os.listdir(f"{args.parent_dir}/{run}"):
            if fn.endswith(".pt"):
                os.remove(f"{args.parent_dir}/{run}/{fn}")

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.max_budget))

if __name__ == '__main__':
    main()