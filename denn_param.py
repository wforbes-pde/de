import pandas as pd
import itertools
import os
import sys
from datetime import datetime
import logging
import numpy as np
#matplotlib.use('Agg')
from denn_helper import NNClass, DEModelClass, MCMCClass, BuildingDataClass, ConcreteDataClass
from denn_helper import return_refine_count, return_layers, return_combo_list
from denn_helper import post_DE, post_DE_MCMC
from denn_matrix import differential_evolution
import random

print_master = False

def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    #logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s %(levelname)s %(message)s')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    start = datetime.now()

    # DE parameter exploration

    application = 'load'
    exercise = 'testing' # validation or testing
    return_ = '5'
    title = f'{application}-{exercise}-{return_}'
    ffs = [ 16,17,18,19,20 ]
    DE_grid = {'G':[ 1000 ], # 
            'NP':[ 4 ] , #
            'F':[ 0.5, ], # 
            'CR': [ 0.7, ], # 
            'mutation_type': [ 'random' ], 
            'NPI': [ 40,  ] , # 
            'track_len': [ 2, ], # min 2 
            'init': [ 'halton', 'he', 'uniform', 'latin'
                    ], 
            'refine_param': [ # (10,10,10,False), 
                             (10,5,5,True), 
                                ], # (100,2,10) refine_gen_start, refine_current_start, refine_mod_start, refine_random
                            # return_combo_list(ffs, 2),
            'F_refine': [ 'default', ], # 'default', 'variable', 'weight_variable',
            'F_delta': [ 0.10 ], #
            'lowerF': [ 0.1 ], # 0.1
            'upperF': [ 0.9, ], # 0.9
            'mutation_refine': [ 'default',], # 'default', 'variable', 'weight_variable',
            'CR_refine': [ 'default', ], # 'default', 'variable', 'weight_variable', 
            'CR_delta': [ 0.1 ], # 
            'lowerCR': [ 0.1 ], # 0.1
            'upperCR': [ 0.9 ], # 0.9
            'mcmc_args': [  (False, False, None), # no MCMC
                         # (True, True, 2), # mcmc multiple chain
                           # (True, True, 4),
                             (True, False, 1) # mcmc single chain
                         ], # run_mcmc, run_multiple, num_chains
            'burn_in': [ 100, ], 
            'error_dist': [ 'norm', ], # norm, unif
            'error_std': [ 1e-4, ], # 1e-4,
            'fitness_metric': [ 'rmse', ] , # 'rmse', 'r2',
            'run_enh':  [ # run_svd, run_cluster, run_local
                              (False, False, False),
                               (True, True, True),
                            ],
                          # return_combo_list([True, False], 3),
            'layers':  [  #                     
                              (2,10,10), 
                          #None,
                               ], # n1,n2,n3
                       #          return_combo_list(ffs, 3),
            'pred_post_sample': [ 'default', ], 
            'seed': [ None, ], # None for random, int
    }
    
    a = DE_grid.values()
    combinations = list(itertools.product(*a))
    
    # load forecasting
    
    if application == 'load':

        building = 'PepperCanyon'
        load_path = rf'../data/{building}.csv'
        weather_path = r'../data/SanDiegoWeather_all.csv'
        x_cols = ['tmpf', 'dwpf', 'relh', 'sknt', 'feel', 'alti', 'vsby', ]
        ycol = 'kWh'
        daytype = 'weekday'
        Data = BuildingDataClass(building, x_cols, ycol, load_path, weather_path, daytype)
        Data.set_application(application)
        Data.set_exercise(exercise)

    if application == 'concrete':
        daytype=None
        concrete_path = rf'/home/wesley/data/concrete/Concrete_Data.ods'
        x_cols = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 'CoarseAggregate', 'FineAggregate', 'Age']
        ycol = 'Strength'
        Data = ConcreteDataClass(x_cols, ycol, concrete_path)
        Data.set_application(application)
        Data.set_exercise(exercise)

    models = []
    master = []
    run = 0
    result_ids = []
    for param in combinations:
        if True:
        #try:
            logging.critical(f'Starting DE exploration Run {run}/{len(combinations)} {param}')

            G = param[0] # max number of generations
            NP = param[1] # number of parameter vectors in each generation         
            F = param[2] # mutate scaling factor
            CR = param[3] # crossover rate 
            mutation_type = param[4]
            NPI = param[5]
            track_length = param[6]
            init = param[7]
            refine_param = param[8]
            F_refine = param[9]
            F_delta = param[10]
            lowerF = param[11]
            upperF = param[12]
            mutation_refine = param[13]
            CR_refine = param[14]
            CR_delta = param[15]
            lowerCR = param[16]
            upperCR = param[17]              
            mcmc_args = param[18]
            burn_in = param[19]
            error_dist = param[20]
            error_std = param[21]                
            fitness_metric = param[22]          
            run_enh = param[23]       
            layers = param[24]
            pred_post_sample = param[25]
            seed_ = param[26]

            DE_model = DEModelClass(NP, G, F, CR, mutation_type, NPI, init, track_length,
                                    F_refine, F_delta, lowerF, upperF,
                                    mutation_refine, refine_param, 
                                    CR_refine, CR_delta, lowerCR, upperCR,
                                    fitness_metric, run_enh, seed_)
                                
            # checks

            if burn_in >= G: 
                logging.info(f'skipping NP {NP} burn_in {burn_in}')
                continue
            
            # MCMC Class

            MCMC = MCMCClass(mcmc_args, burn_in, error_dist, error_std, run, pred_post_sample)
            
            # neural network structure

            num_layers = 3
            if layers is None:
                layers = return_layers(application,daytype,MCMC)
            n1,n2,n3 = layers

            m = len(Data.x_cols) # feature dimension
            NN_model = NNClass(num_layers, n1, n2, n3, m)

            # run DE and return resultant best-fitting candidate

            DE_model.set_run(run)
            if seed_ == None:
                run_seed = random.randint(0, 5000)
                DE_model.set_seed(run_seed)
                np.random.seed(DE_model.seed)
            else:
                DE_model.set_seed(seed_)
                np.random.seed(DE_model.seed)
            optimum_point, gen_points, dfs, mcmc_chain, scaler, X_train_scaled, y_train = differential_evolution(DE_model, NN_model, Data, MCMC)

            # determine validation or testing data set

            if Data.exercise in ['validation']:        
                X_val = Data.X_val
                y_val = Data.y_val
                y_test = y_val
                test_x = scaler.transform(X_val)
                test_data = None

            if Data.exercise in ['testing']:
                if Data.application in ['concrete']:
                    test_x = Data.X_test
                    y_test = Data.y_test
                    test_x = scaler.transform(test_x)
                    test_data = None
                
                if Data.application in ['load']:
                    X_test = Data.X_test
                    y_test = Data.y_test
                    test_x = scaler.transform(X_test)
                    test_data = None

            NP_indices = list(np.arange(0,NP))

            if not MCMC.run_mcmc:
                args2 = optimum_point, gen_points, dfs, scaler, X_train_scaled, y_train, y_test
                post_de_args = application, daytype, num_layers, NN_model, Data, test_data, test_x, fitness_metric,\
                        models,ycol, DE_model, NP_indices, print_master, NP, MCMC, DE_model
                models, data = post_DE(post_de_args, args2)
            
            if MCMC.run_mcmc:
                args2 = optimum_point, gen_points, dfs, scaler, X_train_scaled, y_train, y_test
                post_de_mcmc_args = application, daytype, num_layers, NN_model, Data, test_data, test_x, fitness_metric,\
                        ycol, DE_model, NP_indices, print_master, NP, MCMC, DE_model
                models = post_DE_MCMC(post_de_mcmc_args, args2, mcmc_chain,G, MCMC, models)
                
            master.append(dfs)
            run = run + 1

        #except:
         #   run = run + 1
            logging.info(f'error run {run}')                


    # collect results

    full_data = pd.concat(master, sort=False)
    mdata = pd.concat(models, sort=False)

    # runtime

    time_taken = datetime.now() - start
    print(f'runtime was {time_taken}')
    
    # output full data

    kfc = (f'DE-NN-{application}', 'refinement')
    out_dir = r'../output'
    output_name = '-'.join(kfc)
    output_loc = os.path.join(out_dir + os.sep + output_name + '.csv')
    logging.critical(f'Saving to {output_loc}')
    full_data.to_csv(output_loc)
    
    # slice exit generation

    for_agg = full_data[full_data['Exit'] == 'True' ].copy()

    # summary standard average

    key = ['G', 'NP', 'NPI', 'F', 'CR', 'mutation_type',
           'F_delta', 'lowerF', 'upperF', 'F_refine', 'refine_param', 
           'mutation_refine', 'lowerCR', 'upperCR', 'CR_refine', 'CR_delta', 
           'run_mcmc', 'burn_in',  'track_len', 'pred_post_sample',
           'fitness_metric', 'run_enh', 'error_dist', 'error_std', 'init', 
           'layers',  ] # add neurons for 1 layer ADD BACK 'c'
    if MCMC.run_mcmc:
        key_cols = [f'Test_{fitness_metric}_MCMC_mode', 'Test_RMSE_MCMC_mode', 'TestStd', 'Test_RMSE_MCMC_mean']
        #key.remove('c')
    else:
        key_cols = [f'Test_{fitness_metric}', 'Test_RMSE', 'TestStd']
    
    # across each group and index c
    mdata['c'] = mdata['c'].astype(str)

    perf_m = mdata.groupby(key)[key_cols].aggregate(['mean','count', 'min', 'max', 'std'])
    perf_m = perf_m.reset_index(drop=False)

    # what about minimum across a run irrespective of index?
    # i.e. minimum at c=1 one run, then minimum at c=2 at the end.

    key2 = key.copy()
    #key2.remove('c')
    key2.append('Run')
    key_cols2 = key_cols.copy()
    key_cols2.remove('TestStd') # add this?
    perf_m2 = mdata.groupby(key2)[key_cols2].aggregate(['mean','count', 'min', 'max', 'std'])
    perf_m2 = perf_m2.reset_index(drop=False)

    discard = ['F_W0', 'F_W1', 'F_W2', 'F_W3', 'F_b0', 'F_b1', 'F_b2', 'F_b3',
                        'F2_W0', 'F2_W1', 'F2_W2', 'F2_W3', 'F2_b0', 'F2_b1', 'F2_b2','F2_b3']
    mdata = mdata[mdata.columns[~mdata.columns.isin(discard )]]

    # merge columns

    perf_m.columns = perf_m.columns.map('_'.join)
    perf_m2.columns = perf_m2.columns.map('_'.join)

    # refine count

    ref_count = return_refine_count(full_data)

    # group minimum

    if MCMC.run_mcmc:
        perf_m['Test_RMSE_MCMC_mode_mean'] = pd.to_numeric(perf_m['Test_RMSE_MCMC_mode_mean'])
        cols = ['NP_', 'G_', 'fitness_metric_', ]
        test = perf_m.loc[perf_m.groupby(cols)['Test_RMSE_MCMC_mode_mean'].idxmin()]

    if not MCMC.run_mcmc:
        perf_m['Test_RMSE_mean'] = pd.to_numeric(perf_m['Test_RMSE_mean'])
        cols = ['NP_', 'G_', 'fitness_metric_',]
        test = perf_m.loc[perf_m.groupby(cols)['Test_RMSE_mean'].idxmin()]

    # output summary data

    kfc = (f'DE-NN', f'{title}', f'{daytype}') 
    out_dir = r'../output'
    output_name = '-'.join(kfc)
    output_loc = os.path.join(out_dir + os.sep + output_name + '.ods')
    logging.critical(f'Saving to {output_loc}')

    with pd.ExcelWriter(output_loc ) as writer:
        for_agg.to_excel(writer, sheet_name = 'training_exit', index=False)
        mdata.to_excel(writer, sheet_name = 'model_data', index=False)
        perf_m.to_excel(writer, sheet_name = 'model_data_mean')
        test.to_excel(writer, sheet_name = 'model_data_mean_min')
        ref_count.to_excel(writer, sheet_name = 'refine_count')
        perf_m2.to_excel(writer, sheet_name = 'test')

if __name__ == '__main__':
    main(sys.argv[1:])
    
