# noisy-eval-in-fl
[![DOI](https://zenodo.org/badge/539070342.svg)](https://zenodo.org/badge/latestdoi/539070342)

Repository to accompany "On Noisy Evaluation in Federated Hyperparameter Tuning". To appear in MLSys2023.


## Setup

In a Linux terminal:
1. Pull this repository by running `git clone https://github.com/imkevinkuo/noisy-eval-in-fl`.
2. Install miniconda3 from https://docs.conda.io/en/latest/miniconda.html.
3. Set up the conda environment with name `noisyfl`. Run the following command:
```conda env create -n noisyfl -f environment.yml```
4. Activate the environment with `conda activate noisyfl`.

`torch.cuda` requires CUDA-compatible hardware, CUDA Toolkit, and potentially manufacturer drivers. We ran our experiments with 8 NVIDIA GeForce GTX 1080 Ti GPUs and CUDA Toolkit 11.6. The CUDA installer can be downloaded from https://developer.nvidia.com/cuda-11-6-0-download-archive.

## Data Preparation

**We provide preprocessed data at** https://drive.google.com/file/d/1iDK5JvEiv3Vz0jNV05cNGKBGbclBwQNu/view?usp=sharing. **Place** `data.tar.gz` **in the project directory and extract the contents with** `tar -xzvf data.tar.gz`. **No other steps are needed.**

`notebooks/generate_data.ipynb` contains code to set up the datasets from scratch, but does not need to be run.

## Model Training

**We provide training logs for CIFAR10 and FEMNIST so artifact reviewers may test the analysis notebook without depending on the functionality of training. The file** `runs.tar.gz` **is included in the repository. Extract the contents with** `tar -xzvf runs.tar.gz`.

Training uses either a weighted or uniform loss across clients. A uniform evaluation bounds individual client sensitivity, and therefore we analyze configurations trained on a uniformly weighted loss runs whenever differential privacy. However, in all other experiments, we weight the losses and evaluations by client size.

The main scripts are named `fedtrain_simple.py`, `fedtrain_tpe.py`, and `fedtrain_bohb.py`.

`fedtrain_simple.py` trains a single configuration which is later post-processed in the notebook analysis of RS and HB. The optimization hyperparameters are passed as arguments into this script.

`fedtrain_tpe.py` and `fedtrain_bohb.py` run the respective HP tuning algorithms and depend on `fedtrain_simple.py` for model training and evaluation. The two main arguments `--batch` and `--eps` are used to set the number of subsampled evaluation clients and the value of epsilon used for differentially private evaluation.

Each `fedtrain_*.py` script has a corresponding `init_*.py` wrapper. This wrapper runs multiple trials of the corresponding `fedtrain` script. 

**Running each of the `init` scripts once without any modifications will produce analysis results for CIFAR10.** The current scripts synchronously execute all trials on a single GPU. The number of trials and run-to-GPU assignment can be changed by modifying the `init` files..

```
python init_simple.py
python init_tpe.py
python init_bohb.py
python init_tpe_noisy.py
python init_bohb_noisy.py
```
## Results and Logging

After model training, several nested logging directories are created. These correspond to configurations of: the training objective (losses weighted uniformly or by client data size), HPO wrapper (none, TPE, or BOHB), and dataset.
```
runs_unif, runs_weighted
    runs_simple, runs_tpe, runs_bohb
        runs_cifar10, runs_femnist, runs_stackoverflow, runs_reddit
            train, eval
                run0, run1, ...
```

`train` and `eval` store the information for individual training runs i.e. a single HP configuration. `train/{run_name}` stores the TensorBoard `.events` file and and arguments `args.json`, while `eval/{run_name}` stores a list of evaluated client error rates in a pickle file with the name `P{X}_R(Y).pkl`. `X` is the probability for the three data heterogeneity settings (0, 0.5, 1) and `Y` is the round number (by default, we eval every 15 rounds and train for a total of 405).



## Analysis

**Results from the init scripts are logged in the following directories:**

```
runs_weighted/runs_simple/runs_cifar10
runs_weighted/runs_simple/runs_femnist
runs_unif/runs_simple/runs_cifar10
runs_unif/runs_simple/runs_femnist
runs_unif/runs_bohb/runs_cifar10
runs_unif/runs_bohb/runs_cifar10_s_e=100
runs_unif/runs_tpe/runs_cifar10
runs_unif/runs_tpe/runs_cifar10_s_e=100
```

Run all the cells in the notebook `analysis.ipynb`. The plots will be displayed in the notebook.
