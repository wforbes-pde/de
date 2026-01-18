# eDE-BNN
#### Versions: 0.0.1
#### Last update: Oct 30, 2025

An enhanced differential evolution-Bayesian neural network (DE-BNN) algorithm for regression problems.

## Prerequisites

It is highly recommend to utilize `miniconda` or `pip` package managers to prevent unforeseen dependency conflicts.

- Python: => 3.10.15
- Numpy: => 1.26.4
- Scipy: => 1.14.1
- Matplotlib: > 3.9.2
- scikit-learn: => 1.5.2


It is highly recommend to create a new environment in `miniconda` to run DE-BNN to prevent packages conflicts.

## Dependencies

DE-BNN requires the following dependencies to run:

        # Create new anaconda environment
        conda create -y --name debnn python=>3.10.15
        conda activate debnn
        conda install -y "numpy=>1.26.4" "scipy=>1.14.1" "matplotlib=>3.9.2" scikit-learn pandas

## Installations

To install DE-BNN simply clone the repo:

        git clone https://github.com/wforbes-pde/de.git
        cd de

## Usage

DE-BNN is presently setup only for a 3-layer MLP.

denn_param.py

    Create a data Class based on your problem - needs X_train and y_train.
    Specify parameters in grid search. 

denn_matrix.py

    Contains differential evolution and selection functions. 

denn_helper.py

    Update the fitness() function definition in DEModelClass as appropriate.
    
Here are the grid search parameters. 

    G: maximum generation.
    NP: number of population candidates. 
    F: default mutation scaling factor. 
    CR default crossover value.
    mutation_type: default mutation type.
    NPI: number of initial population candidates.
    track_len: number of generations in running-average residual; min 2.
    init: type of initial population generation; options: 'halton', 'he', 'uniform', 'latin'
    refine_param: enhancement parameters; (refine_gen_start, refine_current_start, refine_mod_start, refine_random).
    F_refine: type of F value enhancement. options: 'default', 'variable', 'weight_variable',
    F_delta: interval of points in discrete distribution to sample from.
    lowerF: lower bound of F discrete distribution.
    upperF: upper bound of F discrete distribution.
    mutation_refine: type of mutation type value enhancement. options: 'default', 'variable', 'weight_variable',
    CR_refine: type of CR value enhancement. options: 'default', 'variable', 'weight_variable',
    CR_delta:  interval of points in discrete distribution to sample from.
    lowerCR: lower bound of CR discrete distribution.
    upperCR: upper bound of CR discrete distribution.
    mcmc_args: arguments for MCMC, (run_mcmc, run_multiple_chains, num_chains), e.g. (True,False,1).
    burn_in: number of MCMC burn in generations.
    error_dist: distribution type for error distribution for mutation error; used to satisfy detailed balance. options: 'norm', 'unif'.
    error_std: distribution mean for error distribution for mutation error; used to satisfy detailed balance.
    fitness_metric: fitness function metric. options: 'rmse', 'r2', 'mae', 'mape', 'mse'
    run_enh: trigger for each enhancement method (run_svd, run_cluster, run_local).
    layers: number of neurons in each of the 3 MLP layers (n1,n2,n3).
    pred_post_sample: number of samples from the end to use. default uses length of chain.
    seed: set numpy seed; None for random int for each parameter set run. int for a fixed seed.


## Update

DE-BNN is under active development and welcomes input.


## Citation

TBD.
