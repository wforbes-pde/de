import numpy as np
import pandas as pd
import random
import logging
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler  
from datetime import datetime
from pyDOE import lhs

print_master = False

# Function to selectively clip arrays
def selective_clip(arr):
    # Define a threshold for clipping
    threshold = 500  # Adjust based on your data scale
    clip_min = -500
    clip_max = 500
    # Check if max absolute value exceeds threshold
    if np.max(np.abs(arr)) > threshold:
        return np.clip(arr, clip_min, clip_max).astype(np.float32)
    return arr.astype(np.float32)

# https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube
def lhs_init(samples_, dim, bounds):

    # appears to be a samples of R^b
    # is this okay? 

    sample = lhs(n=dim, samples=samples_)
    pop = np.zeros((samples_, dim))
    for i in range(dim):
        pop[:, i] = sample[:, i] * bounds
    return pop


def convert_biases(NP, n_, n1,n2,n3,initial_NP_indices,
                   x_b0,x_b1,x_b2,x_b3):

    b0_ = np.repeat(x_b0, n_, axis=1)
    b1_ = np.repeat(x_b1, n_, axis=1)
    b2_ = np.repeat(x_b2, n_, axis=1)
    b3_ = np.repeat(x_b3, n_, axis=1)
    
    return b0_, b1_, b2_, b3_


def setup_gen(m,n1,n2,n3,num_samples):
    x_W0 = np.full((num_samples, m, n1), 1.0, dtype=np.float32)
    x_W1 = np.full((num_samples, n1, n2), 1.0, dtype=np.float32)
    x_W2 = np.full((num_samples, n2, n3), 1.0, dtype=np.float32)
    x_W3 = np.full((num_samples, n3, 1), 1.0, dtype=np.float32)

    x_b0 = np.full((num_samples, 1, n1), 1.0, dtype=np.float32)
    x_b1 = np.full((num_samples, 1, n2), 1.0, dtype=np.float32)
    x_b2 = np.full((num_samples, 1, n3), 1.0, dtype=np.float32)
    x_b3 = np.full((num_samples, 1, 1), 1.0, dtype=np.float32)
    return x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3 

def generate_initial_population(a, b, NP_indices, key, init):
    
    # bias matrix initialization based on 
    # a number of rows, n_input
    # b number of columns, n_output
    # NP number of candidates

    NP = len(NP_indices)    

    # Kaiming He weight initialization for relu
    # loc = mean, scale = standard deviation    

    #candidates = {}

    if init == 'latin':
        bounds=1
        candidates = np.zeros((NP,a,b))
        x = lhs_init(len(NP_indices), a*b, bounds)
        for j in NP_indices:
            trans = x[j,:].reshape(len(x[j,:]),1)
            xx = trans.reshape(a,b)
            candidates[j,:,:] = xx.astype(np.float32)

    if init == 'halton':
        candidates = np.zeros((NP,a,b))
        sampler = qmc.Halton(d=a*b, scramble=True)
        x = sampler.random(n=len(NP_indices))
        for j in NP_indices:
            trans = x[j,:].reshape(len(x[j,:]),1)
            xx = trans.reshape(a,b)
            candidates[j,:,:] = xx.astype(np.float32)

    # he, uniform

    if init == 'he':
        mean_ = 0
        sigma = np.sqrt(2.0) * np.sqrt(2 / (a+b))
        x = np.random.normal(loc=mean_, scale=sigma, size=(NP,a,b))
        candidates = x.astype(np.float32)

    if init == 'uniform':
        r = np.sqrt(2.0) * np.sqrt(6 / (a+b))
        x = np.random.uniform(low=-r, high=r, size=(NP,a,b))
        candidates = x.astype(np.float32) 

    
    return candidates

# X_train, y_train,
def selection(NP_indices, x_dict, y_dict, gen, mindex, 
              current, fitness_metric_, 
              W0, W1, W2, W3, b0, b1, b2, b3,
              x_points, z_points, MCMC, NN_model, DE_model):
    
    x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3 = x_points
    z_W0, z_W1, z_W2, z_W3, z_b0, z_b1, z_b2, z_b3 = z_points
    
    # determine survival of target or trial vector 
    # into the next generation
    
    n_ = len(y_dict)
    X_train = x_dict
    y_train = y_dict
        
    i_accept = 0

    for j in NP_indices:        
        zfit, zyb = DE_model.fitness(X_train, z_W0[j], z_W1[j], z_W2[j], z_W3[j], z_b0[j], z_b1[j], z_b2[j], z_b3[j], y_train, n_, fitness_metric_, NN_model)
        xfit, xyb = DE_model.fitness(X_train, x_W0[j], x_W1[j], x_W2[j], x_W3[j], x_b0[j], x_b1[j], x_b2[j], x_b3[j], y_train, n_, fitness_metric_, NN_model)

        if zfit <= xfit:
            
            x_W0[j] = z_W0[j].copy()
            x_W1[j] = z_W1[j].copy()
            x_W2[j] = z_W2[j].copy()
            x_W3[j] = z_W3[j].copy()
            x_b0[j] = z_b0[j].copy()
            x_b1[j] = z_b1[j].copy()
            x_b2[j] = z_b2[j].copy()
            x_b3[j] = z_b3[j].copy()
            i_accept = i_accept + 1 
        
        # MCMC acceptance

        # likelihood ratio
        # uniform random number alpha

        run_mcmc = MCMC.run_mcmc
        burn_in = MCMC.burn_in
        ratio = np.minimum(1,xfit/zfit)
        alpha = random.uniform(0,1)

        # serial chain

        if run_mcmc and not MCMC.multiple_chain and j == mindex and gen > burn_in:
            W0, W1, W2, W3, b0, b1, b2, b3 = MCMC.serial_chain_MCMC(gen, xfit, zfit, mindex, run_mcmc, ratio, burn_in, j,
                                                                    MCMC, alpha, x_points, z_points, 
                                                                    W0, W1, W2, W3, b0, b1, b2, b3)
            
        if run_mcmc and MCMC.multiple_chain and j in MCMC.top_chains and gen > burn_in:
            
            # for each index chain

            W0, W1, W2, W3, b0, b1, b2, b3 = MCMC.multiple_chain_MCMC(gen, xfit, zfit, mindex, run_mcmc, ratio, burn_in,j,
                                                                                            MCMC, alpha, x_points, z_points, 
                                                                                            W0, W1, W2, W3, b0, b1, b2, b3)

    selected_points = x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3
    return selected_points, i_accept, W0, W1, W2, W3, b0, b1, b2, b3

def differential_evolution(DE_model, NN_model, Data, MCMC):
    
    NP = DE_model.NP
    G = DE_model.g
    F = DE_model.F
    CR = DE_model.CR
    mutation_type = DE_model.mutation_type  
    fitness_metric = DE_model.fitness_metric
    run_enh = DE_model.run_enh

    # parameters

    F_delta = DE_model.F_delta
    NPI = np.maximum(DE_model.NPI,DE_model.NP)
    init = DE_model.init
    lowerF = DE_model.lowerF
    upperF = DE_model.upperF
    track_len = DE_model.track_length
    refine_gen_start, refine_current_start, refine_mod_start, refine_random = DE_model.refine_param
    F_refine = DE_model.F_refine    
    mutation_refine = DE_model.mutation_refine

    CR_refine = DE_model.CR_refine
    lowerCR = DE_model.lowerCR
    upperCR = DE_model.upperCR
    CR_delta = DE_model.CR_delta

    # NN structure

    m = NN_model.m
    n1 = NN_model.n1
    n2 = NN_model.n2
    n3 = NN_model.n3
    NN_model.set_phase('training')

    if Data.application in ['load']:    
        X_train = Data.X_train
        y_train = Data.y_train

    # concrete

    if Data.application in ['concrete']:
        X_train = Data.X_train
        y_train = Data.y_train

    # change precision to float32

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_train = np.array(y_train,dtype=np.float32)
    y_train = y_train.reshape(len(y_train),1)

    ######

    NN_model.set_ytrain_std(np.std(y_train,axis=0))

    # feature scaling
    # Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data.
    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    
    scaler = StandardScaler()  
    # Don't cheat - fit only on training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # apply same transformation to test data and validation data

    n_ = len(y_train)     
    
    m_,n_ = X_train.shape
    m__,n__ = y_train.shape
    x_dict = np.zeros((NP,m_,n_))
    y_dict = np.zeros((NP,m__,n__))
    x_dict[:,:,:], y_dict[:,:,:] = X_train,y_train

    # enhancement
    
    run_svd, run_cluster, run_local = run_enh  
    
    # MCMC parameters

    run_mcmc = MCMC.run_mcmc
    burn_in = MCMC.burn_in
    error_dist = MCMC.error_dist
    error_std = MCMC.error_std
    run = DE_model.run
    multiple_chain = MCMC.multiple_chain

    W0, W1, W2, W3, b0, b1, b2, b3 = MCMC.setup_mcmc_array(m,n1,n2,n3,MCMC.chains, G)
    
    NP_indices = list(np.arange(0,NP))
    initial_NP_indices = list(np.arange(0,NPI))
    df_list=[]
    accept_list = []

    # training data tracking for refinement

    gen_train_fitness_list = []
    gen_train_resid_list = []

    # validation data tracking for exit

    gen_val_fitness_list = []
    gen_val_resid_list = []

    rmse_ex = np.array([0])
    
    # start DE exploration
    global acceptance_rate
    global a_rate

    skip = refine_mod_start-1
    svd_filter_r = [skip]
    svd_scalar_r = [skip]
    svd_exp_r = [skip]
    cluster_r =[skip]
    local_r = [skip]
    
    d = m*n1 + n1*n2 + n2*n3 + n3 + n1 + n2 + n3 + 1
    print(f'dimensionality is {d}')
    exit_criteria = False
    i=0

    while not exit_criteria:

        # generate initial population for each weight matrix

        if i == 0:
            logging.info(f'run {run} gen {i} initial population start {DE_model.init}')
            
            ix_W0 = generate_initial_population(m, n1, initial_NP_indices, 'W0', init)
            ix_W1 = generate_initial_population(n1, n2, initial_NP_indices, 'W1', init)
            ix_W2 = generate_initial_population(n2, n3, initial_NP_indices, 'W2', init)
            ix_W3 = generate_initial_population(n3, 1, initial_NP_indices, 'W3', init)

            ix_b0 = generate_initial_population(1, n1, initial_NP_indices, 'b0', init)
            ix_b1 = generate_initial_population(1, n2, initial_NP_indices, 'b1', init)
            ix_b2 = generate_initial_population(1, n3, initial_NP_indices, 'b2', init)
            ix_b3 = generate_initial_population(1, 1, initial_NP_indices, 'b3', init)
            
            # initial population fitness

            initial_fitness = []
            n_ = len(y_train)

            start = datetime.now()
            time_taken = datetime.now() - start    

            for j in initial_NP_indices:
                init_rmse, iyb = DE_model.fitness(X_train, ix_W0[j], ix_W1[j], ix_W2[j], ix_W3[j], ix_b0[j], ix_b1[j], ix_b2[j], ix_b3[j], 
                                  y_train, n_, fitness_metric, NN_model) 
                initial_fitness.append(init_rmse)           
            
            # need to repeat/convert biases into 3d array

            # b0_,b1_,b2_,b3_ = convert_biases(NP, n_, n1,n2,n3,initial_NP_indices,
            #                 ix_b0,ix_b1,ix_b2,ix_b3)
            
            # print(f'b0_ shape {b0_.shape}')
            # print(f'b1_ shape {b1_.shape}')
            # print(f'b2_ shape {b2_.shape}')
            # print(f'b3_ shape {b3_.shape}')

            # print(f'ix_W0_ shape {ix_W0.shape}')
            # print(f'ix_W1_ shape {ix_W1.shape}')
            # print(f'ix_W2_ shape {ix_W2.shape}')
            # print(f'ix_W3_ shape {ix_W3.shape}')

            # rmse_np = DE_model.feed(X_train, ix_W0, ix_W1, ix_W2, ix_W3, b0_, b1_, b2_, b3_, n_, NN_model,
            #                        y_train)
            
            ##########
            
            iidx = np.argpartition(initial_fitness, NP-1)[:NP]
            imin_value = np.amin(initial_fitness)
            imindex = np.where(initial_fitness == imin_value)
            imindex = imindex[0][0]
            mindex = imindex

            # populate initial generation with best candidates

            x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3 = setup_gen(m,n1,n2,n3,NP)

            # looks like indicing this way defaults to "highest" level, i.e. 3d

            for k in NP_indices:
                x_W0[k] = ix_W0[iidx[k]]
                x_W1[k] = ix_W1[iidx[k]]
                x_W2[k] = ix_W2[iidx[k]]
                x_W3[k] = ix_W3[iidx[k]]

                x_b0[k] = ix_b0[iidx[k]]
                x_b1[k] = ix_b1[iidx[k]]
                x_b2[k] = ix_b2[iidx[k]]
                x_b3[k] = ix_b3[iidx[k]]
            
            # set gen best

            gen_best_x_W0 = ix_W0[imindex].copy()
            gen_best_x_W1 = ix_W1[imindex].copy()
            gen_best_x_W2 = ix_W2[imindex].copy()
            gen_best_x_W3 = ix_W3[imindex].copy()

            gen_best_x_b0 = ix_b0[imindex].copy()
            gen_best_x_b1 = ix_b1[imindex].copy()
            gen_best_x_b2 = ix_b2[imindex].copy()
            gen_best_x_b3 = ix_b3[imindex].copy()

            initial_fitness.sort()
            w = np.mean(initial_fitness[:NP])
            logging.info(f'gen {i} best fitness is {initial_fitness[0]}, avg fitness {w}, NPI {NPI}')
            #logging.info(f'gen {i} best fitness is {init_rmse[imindex]}, avg fitness {w}, NPI {NPI}')

            min_value = 500
            
        if i > 0:

            x_W0 = xgp_W0.copy()
            x_W1 = xgp_W1.copy()
            x_W2 = xgp_W2.copy()
            x_W3 = xgp_W3.copy()

            x_b0 = xgp_b0.copy()
            x_b1 = xgp_b1.copy()
            x_b2 = xgp_b2.copy()
            x_b3 = xgp_b3.copy()

            gen_best_x_W0 = gb_W0.copy()
            gen_best_x_W1 = gb_W1.copy()
            gen_best_x_W2 = gb_W2.copy()
            gen_best_x_W3 = gb_W3.copy()

            gen_best_x_b0 = gb_b0.copy()
            gen_best_x_b1 = gb_b1.copy()
            gen_best_x_b2 = gb_b2.copy()
            gen_best_x_b3 = gb_b3.copy()

        if i < track_len:
            train_run_avg_residual_rmse = -1

        if train_run_avg_residual_rmse < 0:
            current = 0

        # mutation parameters

        F = DE_model.F
        CR = DE_model.CR
        mutation_type = DE_model.mutation_type
        lowerF = DE_model.lowerF
        upperF = DE_model.upperF
        F_delta = DE_model.F_delta
        fitness_metric = DE_model.fitness_metric

        # default CR

        CR_W0, CR_W1, CR_W2, CR_W3, CR_b0, CR_b1, CR_b2, CR_b3 = DE_model.return_F_CR(DE_model.CR_refine, lowerF, upperF, CR_delta, DE_model.CR, DE_model, NN_model)

        # default F values for each weight matrix constant

        F_W0, F_W1, F_W2, F_W3, F_b0, F_b1, F_b2, F_b3 = DE_model.return_F_CR(DE_model.F_refine, lowerF, upperF, F_delta, DE_model.F, DE_model, NN_model)
        F2_W0, F2_W1, F2_W2, F2_W3, F2_b0, F2_b1, F2_b2, F2_b3 = DE_model.return_F_CR(DE_model.F_refine, lowerF, upperF, F_delta, DE_model.F, DE_model, NN_model)
        F3_W0, F3_W1, F3_W2, F3_W3, F3_b0, F3_b1, F3_b2, F3_b3 = DE_model.return_F_CR(DE_model.F_refine, lowerF, upperF, F_delta, DE_model.F, DE_model, NN_model)

        F_one = F_W0, F_W1, F_W2, F_W3, F_b0, F_b1, F_b2, F_b3
        F_two = F2_W0, F2_W1, F2_W2, F2_W3, F2_b0, F2_b1, F2_b2, F2_b3
        F_three = F3_W0, F3_W1, F3_W2, F3_W3, F3_b0, F3_b1, F3_b2, F3_b3

        # default mutation type
        
        mutation_list = DE_model.return_mutation_list(NP)    
        mutation_op = DE_model.return_mutation_type(DE_model.mutation_refine, mutation_list, DE_model.mutation_type)
        mutation_W0, mutation_W1, mutation_W2, mutation_W3, mutation_b0, mutation_b1, mutation_b2, mutation_b3 = mutation_op

        # refinement steps

        if train_run_avg_residual_rmse >= 0 and i >= track_len-1:

            current = current + 1
            
            # randomly selection F, mutation, and crossover variation scheme under refinement
            
            if current > refine_current_start and i > refine_gen_start and refine_random:
                
                variation_list = ['default', 'variable', 'weight_variable', 'general']  
                F_refine = random.choice(variation_list)
                variation_list = ['default', 'variable', 'weight_variable'] 
                CR_refine = random.choice(variation_list)
                mutation_refine = random.choice(variation_list)

            if current > refine_current_start and i > refine_gen_start:

                F_W0, F_W1, F_W2, F_W3, F_b0, F_b1, F_b2, F_b3 = DE_model.return_F_CR(F_refine, lowerF, upperF, F_delta, F, DE_model, NN_model)
                F2_W0, F2_W1, F2_W2, F2_W3, F2_b0, F2_b1, F2_b2, F2_b3 = DE_model.return_F_CR(F_refine, lowerF, upperF, F_delta, F, DE_model, NN_model)
                F3_W0, F3_W1, F3_W2, F3_W3, F3_b0, F3_b1, F3_b2, F3_b3 = DE_model.return_F_CR(F_refine, lowerF, upperF, F_delta, DE_model.F, DE_model, NN_model)

                F_one = F_W0, F_W1, F_W2, F_W3, F_b0, F_b1, F_b2, F_b3
                F_two = F2_W0, F2_W1, F2_W2, F2_W3, F2_b0, F2_b1, F2_b2, F2_b3
                F_three = F3_W0, F3_W1, F3_W2, F3_W3, F3_b0, F3_b1, F3_b2, F3_b3

            if current > refine_current_start and i > refine_gen_start:
                CR_W0, CR_W1, CR_W2, CR_W3, CR_b0, CR_b1, CR_b2, CR_b3 = DE_model.return_F_CR(CR_refine, lowerF, upperF, CR_delta, DE_model.CR, DE_model, NN_model)

            if current > refine_current_start and i > refine_gen_start:
                
                mutation_op = DE_model.return_mutation_type(mutation_refine, mutation_list, DE_model.mutation_type)             
                mutation_W0, mutation_W1, mutation_W2, mutation_W3, mutation_b0, mutation_b1, mutation_b2, mutation_b3 = mutation_op
        
        # mutation
            
        y_W0 = DE_model.mutation(NP, NP_indices, F_one, F_two, F_three, x_W0, MCMC, gen_best_x_W0, mutation_W0, 'W0')
        y_W1 = DE_model.mutation(NP, NP_indices, F_one, F_two, F_three, x_W1, MCMC, gen_best_x_W1, mutation_W1, 'W1')
        y_W2 = DE_model.mutation(NP, NP_indices, F_one, F_two, F_three, x_W2, MCMC, gen_best_x_W2, mutation_W2, 'W2')
        y_W3 = DE_model.mutation(NP, NP_indices, F_one, F_two, F_three, x_W3, MCMC, gen_best_x_W3, mutation_W3, 'W3')

        y_b0 = DE_model.mutation(NP, NP_indices, F_one, F_two, F_three, x_b0, MCMC, gen_best_x_b0, mutation_b0, 'b0')
        y_b1 = DE_model.mutation(NP, NP_indices, F_one, F_two, F_three, x_b1, MCMC, gen_best_x_b1, mutation_b1, 'b1')
        y_b2 = DE_model.mutation(NP, NP_indices, F_one, F_two, F_three, x_b2, MCMC, gen_best_x_b2, mutation_b2, 'b2')
        y_b3 = DE_model.mutation(NP, NP_indices, F_one, F_two, F_three, x_b3, MCMC, gen_best_x_b3, mutation_b3, 'b3')
                
        # crossover

        z_W0 = DE_model.crossover_broadcast(NP_indices, y_W0, x_W0, CR_W0, 'W0')
        z_W0 = DE_model.crossover_broadcast(NP_indices, y_W0, x_W0, CR_W0, 'W0')
        z_W1 = DE_model.crossover_broadcast(NP_indices, y_W1, x_W1, CR_W1, 'W1')
        z_W2 = DE_model.crossover_broadcast(NP_indices, y_W2, x_W2, CR_W2, 'W2')
        z_W3 = DE_model.crossover_broadcast(NP_indices, y_W3, x_W3, CR_W3, 'W3')

        z_b0 = DE_model.crossover_broadcast(NP_indices, y_b0, x_b0, CR_b0, 'b0')
        z_b1 = DE_model.crossover_broadcast(NP_indices, y_b1, x_b1, CR_b1, 'b1')
        z_b2 = DE_model.crossover_broadcast(NP_indices, y_b2, x_b2, CR_b2, 'b2')
        z_b3 = DE_model.crossover_broadcast(NP_indices, y_b3, x_b3, CR_b3, 'b3')   
        
        # selection
        
        x_points = x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3
        z_points = z_W0, z_W1, z_W2, z_W3, z_b0, z_b1, z_b2, z_b3

        # Apply to x_points and z_points
        x_points = tuple(selective_clip(arr) for arr in x_points)
        z_points = tuple(selective_clip(arr) for arr in z_points)

        if MCMC.run_mcmc and i < MCMC.burn_in:
            MCMC.set_top_chains(None, None)
        
        selected_points, i_accept, W0, W1, W2, W3, b0, b1, b2, b3 = selection(NP_indices, X_train, y_train,
                                                    i, mindex, current, DE_model.fitness_metric, 
                                                    W0, W1, W2, W3, b0, b1, b2, b3,
                                                    x_points, z_points, MCMC, NN_model, DE_model)
        
        xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = selected_points
        accept_list.append(i_accept)

        errors = []

        # need to repeat/convert biases into 3d array

        b0_,b1_,b2_,b3_ = convert_biases(NP, n_, n1,n2,n3,NP_indices,
                        xgp_b0, xgp_b1, xgp_b2, xgp_b3 )        

        rmse_yp = DE_model.feed(X_train, xgp_W0, xgp_W1, xgp_W2, xgp_W3, b0_, b1_, b2_, b3_, n_, NN_model,)        
        gen_train_score = np.sqrt(np.mean((rmse_yp - y_train[None, :, :])**2, axis=(1, 2)))
        #print(gen_train_score)
        errors.append(gen_train_score)

        # determine best generation point

        gen_fitness_values = np.array(errors)
        min_value = np.amin(gen_fitness_values)
        mindex = np.where(gen_fitness_values == min_value)
        mindex = mindex[0][0] # index integer

        # determine worst generation point

        max_value = np.amax(gen_fitness_values)
        maindex = np.where(gen_fitness_values == max_value )
        maindex = maindex[0][0]

        # define generation best
        
        gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3 = xgp_W0[mindex], xgp_W1[mindex], xgp_W2[mindex], xgp_W3[mindex], xgp_b0[mindex], xgp_b1[mindex], xgp_b2[mindex], xgp_b3[mindex]

        # training residual tracking

        gen_train_fitness_list, train_residual, gen_train_resid_list, train_run_avg_residual_rmse = \
            DE_model.return_running_avg_residual(i, min_value, gen_train_fitness_list, gen_train_resid_list, track_len)
        
        # if train_residual > 0:
        #     boo=False
        #     logging.info(f'run {run} gen {i} index {mindex} {fitness_metric} {min_value} train resid {train_residual} val resid {val_residual} current {current}')
        #     breakpoint()
        
        # refinement

        c_min_value = 0
        l_fit = 0
        
        svd_fit = 0
        s_scalar_value = 0
        s_exp_value = 0

        # SVD filter

        comparison_value = min_value
        #comparison_value = max_value

        # clustering

        if current > refine_current_start and i_accept > 0 and run_cluster and current % refine_mod_start in cluster_r:
            gen_points = xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3
            cluster_points, c_min_value = DE_model.perform_clustering(NP, x_dict, y_dict, comparison_value, maindex, DE_model,
                        NN_model, n_, gen_points,i, gen_fitness_values)            
            xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = cluster_points

        # local search

        if current > refine_current_start and i_accept > 0 and run_local and current % refine_mod_start in local_r:
            gen_points = xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3
            search_points, l_fit = DE_model.perform_search(NP, x_dict, y_dict,comparison_value, maindex, DE_model,
                        NN_model, n_, gen_points,i, NP_indices, current, gen_fitness_values)
            xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = search_points

        # filter svd
        
        if current > refine_current_start and i_accept > 0 and run_svd and current % refine_mod_start in svd_filter_r:
            gen_points = xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3
            svd_points, svd_fit  = DE_model.perform_svd_filter(NP, x_dict, y_dict, comparison_value, maindex, DE_model,
                            NN_model, n_, gen_points,i, NP_indices, current, gen_fitness_values)
            xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = svd_points
        
        # scalar SVD

        if current > refine_current_start and i_accept > 0 and run_svd and current % refine_mod_start in svd_scalar_r:
            gen_points = xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3
            svd_scalar_points, s_scalar_value  = DE_model.perform_svd_scalar(NP, x_dict, y_dict, comparison_value, maindex, DE_model,
                                    NN_model, n_, gen_points,i, NP_indices, current, gen_fitness_values)
            xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = svd_scalar_points
        
        # exp scalar

        if current > refine_current_start and i_accept > 0 and run_svd and current % refine_mod_start in svd_exp_r:
            gen_points = xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3
            svd_exp_points, s_exp_value  = DE_model.perform_svd_exp(NP, x_dict, y_dict, comparison_value, maindex, DE_model,
                                            NN_model, n_, gen_points,i, NP_indices, current, gen_fitness_values)
            xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = svd_exp_points        
        
        if i < MCMC.burn_in:
            MCMC.set_acceptance_rate(0)
        
        # efficiency test

        if MCMC.run_mcmc:
            MCMC.set_top_chains(gen_fitness_values, MCMC.chains)
        
        # collect parameters and data
        
        # if refine_random:
        #     F_refine = 'random'
        #     CR_refine = 'random'
        #     mutation_refine = 'random'
        
        df = pd.DataFrame({'Run':[DE_model.run], 'Generation':[i], 'F':[F], 'CR':[CR], 'G':[G], 'NP':[NP], 'NPI':[NPI],'mutation_type':[DE_model.mutation_type],
                        'lowerF':[lowerF], 'upperF':[upperF], 'F_delta':[F_delta], 'init':[str(init)], 
                        'refine_param':[str(DE_model.refine_param)], 'F_refine':[F_refine], 'mutation_refine':[mutation_refine], 
                        'lowerCR':[lowerCR], 'upperCR':[upperCR], 'CR_refine':[CR_refine], 'CR_delta':[CR_delta],
                        'residual':[train_residual], 'run_avg_residual':[train_run_avg_residual_rmse], 'track_len':[track_len],
                        'mutation_type_':[mutation_W0], 'fitness_metric':[fitness_metric],
                        'run_enh':[str(run_enh)], 'current':[current], 'i_accept':[i_accept], 
                        'run_mcmc':[str(MCMC.run_mcmc_arg)], 'burn_in':[burn_in], 'error_dist':[error_dist], 'error_std':[error_std], 
                        'Acceptance':[ MCMC.acceptance_rate], 'pred_post_sample':[str(MCMC.pred_post_sample)], 'layers':[f'({n1},{n2},{n3})'], 
                        'F_W0':[F_W0], 'F_W1':[F_W1], 'F_W2':[F_W2], 'F_W3':[F_W3], 'F_b0':[F_b0], 'F_b1':[F_b1], 'F_b2':[F_b2], 'F_b3':[F_b3],
                        'F2_W0':[F2_W0], 'F2_W1':[F2_W1], 'F2_W2':[F2_W2], 'F2_W3':[F2_W3], 'F2_b0':[F2_b0], 'F2_b1':[F2_b1], 'F2_b2':[F2_b2], 'F2_b3':[F2_b3],                         
                        'TrainRMSE':[min_value], 
                        'clustering_score':[c_min_value], 'local_score':[l_fit], 
                        'svd_value':[svd_fit], 's_scalar_value':[s_scalar_value], 's_exp_value':[s_exp_value], 
                        'seed':[DE_model.seed], 'Exit':[str(False)],
                    })        
        
        logging.info(f'run {run} gen {i} index {mindex} {fitness_metric} {min_value} train resid {train_residual} current {current}')
        
        exit_criteria = i == G-1        
        
        if not exit_criteria:
            df_list.append(df)

        if exit_criteria:
            logging.info(f'run {run} gen {i} exit criteria')
            df['Exit'] = str(True)
            df_list.append(df)
            break
        i=i+1

    if run_mcmc:
        mcmc_chain = W0, W1, W2, W3, b0, b1, b2, b3
        
    if not run_mcmc:    
        mcmc_chain = None

    optimum_point = xgp_W0[mindex], xgp_W1[mindex], xgp_W2[mindex], xgp_W3[mindex], xgp_b0[mindex], xgp_b1[mindex], xgp_b2[mindex], xgp_b3[mindex]
    gen_points = xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3

    dfs = pd.concat(df_list, sort = False)
    dfs['AcceptanceRate'] = np.sum(dfs.Acceptance)/G

    blah = pd.DataFrame(dfs, columns = ['current'])
    blah = blah[blah['current'] > 0].copy()
    dfs['StagnationPerc'] = len(blah)/G

    boo = pd.DataFrame(dfs, columns = ['current'])
    boo['test'] = boo['current'].shift(-1)
    boo.loc[(boo['current'] > 0) & (boo['test'] == 0), 'StagnationLen'] = True
    boo = boo[boo['StagnationLen'] == True].copy()
    stagnation_mean = np.mean(boo.current)
    dfs['MeanStagLen'] = stagnation_mean

    if MCMC.run_mcmc:
        #mciidx = np.argpartition(gen_fitness_values, MCMC.chains-1)[:MCMC.chains]
        MCMC.set_top_chains(gen_fitness_values, MCMC.chains)

    if print_master:
        xcol = 'datetime'
        x_2020 = Data.training[xcol].copy()
        gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3 = optimum_point
        yp = DE_model.DENN_forecast(X_train, gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3, NN_model, MCMC)
        daytype = Data.daytype
        application = Data.application
        label = f'DE-BNN Predicted-{fitness_metric}-{application}'
        file_ext = f'houston-{application}-{daytype}-denn-train-{run}-{mindex}.png'
        DE_model.plot(x_2020, Data.target, y_train, yp, label, file_ext)

    return optimum_point, gen_points, dfs, mcmc_chain, scaler, X_train, y_train 

