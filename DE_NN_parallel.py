import pandas as pd
from math import comb
import itertools
import os
import sys
import ray
from datetime import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from DE_matrix_NN_C import differential_evolution, NNClass, DEModelClass, feed_forward, DENN_forecast
#from DE_matrix_NN_CP import differential_evolution, NNClass, DEModelClass, feed_forward, DENN_forecast
from pandas_ods_reader import read_ods
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

dir_path = os.path.dirname(os.path.realpath(__file__))

start = datetime.now()

def MLP_model(df):

    param_grid = {'hidden_layer_sizes':[ # (20,20), 
                                        (20,20,20), 
                                        ],
            'activation':['relu', ], # 'logistic', 'tanh'
            'solver':['adam', ],  # 'sgd'
            'alpha': [ 1e-2, ], #  1e-5,1e-4, 1e-3, 1e-2, 1e-1
            'learning_rate': ['adaptive'], 
            'max_iter': [250], 
            'batch_size': [32,], #  64,128,256
    }

    a = param_grid.values()
    combinations = list(itertools.product(*a))
    m = len(combinations)
    print(f'combination length {m}')

    y_col = ['LMP']

    X = np.array(df[x_cols])
    y = np.array(df[y_col])

    # split data set

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    models = []
    j=1
    for param in combinations:
        print(f'starting {j}/{m}')
        model, gs = neural_network(X_train, X_test, y_train, y_test, param)
        models.append(model)
        j = j + 1

    results = pd.concat(models, sort=False)

    # output final

    kfc = ('MLP', 'NN') 
    out_dir = r'/home/wesley/repos/rbf-fd/data'
    output_name = '-'.join(kfc)
    output_loc = os.path.join(out_dir + os.sep + output_name + '.ods')

    logging.info(f'Saving to {output_loc}')
    with pd.ExcelWriter(output_loc ) as writer:
        results.to_excel(writer, sheet_name = 'runs', index=False)
    
    if False:
        
        ycol = 'LMP'
        xcol = 'datetime'
        x = lmp[xcol]
        x = df[xcol]

        y_pred = gs.predict(X)

        plt.figure(figsize=(12,6))
        plt.plot(x, y, label='Historical', linewidth=1)
        plt.plot(x, y_pred, color='m', label='NN Predicted', linewidth=1)
        plt.xlabel('datetime')
        plt.ylabel('LMP')
        plt.legend(fontsize = 10, loc='upper left')
        plt.savefig(f'images/houston-lmp-con.png',
                    format='png',
                    dpi=300,
                    bbox_inches='tight')
        plt.show()

    return results, gs

if False:

    kfc = ('DE-NN', 'initial') 
    out_dir = r'/home/wesley/repos/rbf-fd/data'
    output_name = '-'.join(kfc)
    output_loc = os.path.join(out_dir + os.sep + output_name + '.ods')

    # logging.info(f'Saving to {output_loc}')
    # with pd.ExcelWriter(output_loc ) as writer:
    #     pd.DataFrame(W0).to_excel(writer, sheet_name = 'W0', index=False)
    #     pd.DataFrame(W1).to_excel(writer, sheet_name = 'W1', index=False)
    #     pd.DataFrame(W2).to_excel(writer, sheet_name = 'W2', index=False)
    #     pd.DataFrame(W3).to_excel(writer, sheet_name = 'W3', index=False)

def load_training_lmp():

    sheet_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    path = r'/home/wesley/BoiseState/Research/data/rpt.00013060.0000000000000000.DAMLZHBSPP_2022.ods'
    logging.info(f'reading {path}')
    lmp = pd.read_excel(path, sheet_name=None, engine="odf")
    lmp_df = pd.concat(lmp, sort=False)

    lz_list = ['LZ_HOUSTON']
    lz_mask = lmp_df['Settlement Point'].isin(lz_list)
    lmp = lmp_df[lz_mask].copy()
    lmp['Hour Ending'] = lmp['Hour Ending'].str[:2]
    lmp['Hour Ending'] = lmp['Hour Ending'].astype(int)
    lmp['HourStarting'] = lmp['Hour Ending'] - 1
    lmp['Delivery Date'] = pd.to_datetime(lmp['Delivery Date'])
    lmp['datetime'] = lmp['Delivery Date'] + pd.to_timedelta(lmp['HourStarting'], unit = 'h')

    lcols = ['datetime', 'Settlement Point Price']
    lmp = lmp[lcols].copy()
    lmp = lmp.rename(columns={'Settlement Point Price':'LMP'})
    lmp=lmp.reset_index(drop=True)

    return lmp


def load_training_weather(x_cols):

    path = r'/home/wesley/BoiseState/Research/data/IAH.csv'
    logging.info(f'reading {path}')
    weather = pd.read_csv(path)

    #x_cols = ['tmpf', 'dwpf', 'relh', 'sknt', 'feel', 'alti']
    cols = ['valid'] + x_cols
    weather = weather[cols].copy()
    weather=weather.reset_index(drop=True)
    weather['valid'] = pd.to_datetime(weather['valid'])
    weather = weather.resample('H', on='valid').mean()
    weather = weather.reset_index(drop = False)
    weather['datetime'] = pd.to_datetime(weather['valid'])
    del weather['valid']
    weather=weather.dropna()

    return weather


def load_testing_lmp():

    sheet_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    path = r'/home/wesley/BoiseState/Research/data/rpt.00013060.0000000000000000.DAMLZHBSPP_2023.ods'
    logging.info(f'reading {path}')
    lmp = pd.read_excel(path, sheet_name=None, engine="odf")
    lmp_df = pd.concat(lmp, sort=False)

    lz_list = ['LZ_HOUSTON']
    lz_mask = lmp_df['Settlement Point'].isin(lz_list)
    lmp = lmp_df[lz_mask].copy()
    lmp['Hour Ending'] = lmp['Hour Ending'].str[:2]
    lmp['Hour Ending'] = lmp['Hour Ending'].astype(int)
    lmp['HourStarting'] = lmp['Hour Ending'] - 1
    lmp['Delivery Date'] = pd.to_datetime(lmp['Delivery Date'])
    lmp['datetime'] = lmp['Delivery Date'] + pd.to_timedelta(lmp['HourStarting'], unit = 'h')

    lcols = ['datetime', 'Settlement Point Price']
    lmp = lmp[lcols].copy()
    lmp = lmp.rename(columns={'Settlement Point Price':'LMP'})
    lmp=lmp.reset_index(drop=True)

    return lmp

def load_testing_weather(x_cols):

    path = r'/home/wesley/BoiseState/Research/data/IAH2023.csv'
    logging.info(f'reading {path}')
    weather = pd.read_csv(path)

    x_cols = ['tmpf', 'dwpf', 'relh', 'sknt', 'feel', 'alti']
    cols = ['valid'] + x_cols
    weather = weather[cols].copy()
    weather=weather.reset_index(drop=True)
    weather['valid'] = pd.to_datetime(weather['valid'])
    weather = weather.resample('H', on='valid').mean()
    weather = weather.reset_index(drop = False)
    weather['datetime'] = pd.to_datetime(weather['valid'])
    del weather['valid']
    weather=weather.dropna()

    return weather


def neural_network(X_train, X_test, y_train, y_test, param):

    # metric tracking

    model = pd.DataFrame()

    hidden_layer_sizes_ = param[0]
    activation_ = param[1]
    solver_ = param[2]
    alpha_ = param[3]
    learning_rate_ = param[4]
    max_iter_ = param[5]
    batch_size_ = param[6]

    gs = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes_,
                      activation=activation_,
                      solver=solver_,
                      alpha=alpha_,
                      learning_rate=learning_rate_,
                      max_iter=max_iter_,
                      random_state=42,
                      batch_size=batch_size_,
                      early_stopping=True,
                      validation_fraction=0.3                      
                      )

    gs.fit(X_train, y_train.ravel() )
    y_pred = gs.predict(X_test)

    model.loc[0, 'LoadZone'] = 'Houston'

    test_set_rsquared = gs.score(X_test, y_test)
    test_set_rmse = np.sqrt(mean_squared_error(y_test, y_pred) )

    model['solver'] = solver_
    model['max_iter'] = max_iter_
    model['learning_rate'] = learning_rate_
    model['hidden_layer_sizes'] = str(hidden_layer_sizes_)
    model['alpha'] = alpha_
    model['activation'] = activation_
    model['batch_size'] = batch_size_
    model['rsquared'] = test_set_rsquared
    model['rmse'] = test_set_rmse

    return model, gs

# data = np.random.normal(size=(100,10,50))

# out = np.zeros((data.shape[0],data.shape[1],data.shape[1],data.shape[2]))

# for i in range(data.shape[1]):
#     for j in range((data.shape[1])):
#         out[:,i,j,:] = data[:,i,:]-data[:,j,:]

# load LMP data training 2022

lmp = load_training_lmp()

# load weather data training 2022

x_cols = ['tmpf', 'dwpf', 'relh', 'sknt', 'feel', 'alti']
weather = load_training_weather(x_cols)

# merge load and weather training 2022

training = pd.merge(weather, lmp, on = ['datetime'])
x = training[x_cols].copy()

# MLP sklearn

#mlp_results, nn = MLP_model(training)
#sys.exit()

# parameter exploration
# DE parameters

DE_grid = {'g':[30], # 400
            'NP':[200], # 200
           'F':[0.5], # 0.57
           'CR': [0.75], # 0.75
           'angle': [90], # 90
           'mutation_type': ['best'], # 
           'clustering_type': [None], # spectral, agg
           'num_of_clusters': [10], # 
           'cluster_gen_begin': [20], # 
           'rotate_gen_begin': [20], # 
           'num_replace': [2], # 20
           'tol': [-0.1],           
           'NPI': [1600],
           'F_scaling': [0.95,0.9,0.85,0.8],
}

a = DE_grid.values()
combinations = list(itertools.product(*a))

result_list = []
master = []

ray.init(num_cpus=3)

for param in combinations:        
        print(f'Starting parallel DE exploration {param}')
        g = param[0] # max number of generations
        NP = param[1] # number of parameter vectors in each generation         
        F = param[2] # mutate scaling factor
        CR = param[3] # crossover rate 
        angle = param[4]
        mutation_type = param[5]
        clustering_type = param[6]
        num_of_clusters = param[7]
        cluster_gen_begin = param[8]
        rotate_gen_begin = param[9]
        num_replace = param[10]
        tol = param[11]
        NPI = param[12]
        F_scaling = param[13]

        DE_model = DEModelClass(NP, g, F, CR, mutation_type, clustering_type, num_of_clusters, 
                                cluster_gen_begin, num_replace, angle, tol, rotate_gen_begin,
                                NPI, F_scaling)

        # neural network structure

        num_layers = 3
        n1 = 20
        n2 = 20
        n3 = 20
        activation = 'relu'

        # constants

        m = len(x_cols) # feature dimension
        n = len(training) # output dimension

        y_true = training['LMP']

        NN_model = NNClass(x, y_true, num_layers, n1, n2, n3, activation)

        # run DE

        rmse_values = []
        df_all = []        

        #final_gen, dfs, error_values = differential_evolution(DE_model, rmse_values, m, n1, NN_model)

        result_list.append(differential_evolution.remote(DE_model, rmse_values, m, n1, NN_model ) )

results = ray.get(result_list)
results = pd.concat(results, sort=False)
master.append(results)
result_list = []
        
ray.shutdown()
data = pd.concat(master, sort=False)

time_taken = datetime.now() - start
logging.info(f'runtime was {time_taken}')

# output

kfc = ('exploration', 'DE-NN' ) 
out_dir = r'/home/wesley/repos/rbf-fd/data'
output_name = '-'.join(kfc)
output_loc = os.path.join(out_dir + os.sep + output_name + '.ods')

logging.info(f'Saving to {output_loc}')
with pd.ExcelWriter(output_loc ) as writer:
    data.to_excel(writer, sheet_name = 'data', index=False)

# W0 = final_gen[0]
# W1 = final_gen[1]
# W2 = final_gen[2]
# W3 = final_gen[3]
# b0 = final_gen[4]
# b1 = final_gen[5]
# b2 = final_gen[6]
# b3 = final_gen[7]

# rmse_values = []
# rmse = feed_forward(x, W0, W1, W2, W3, b0, b1, b2, b3, y_true, activation)
# print(f'final rmse {rmse}')

# time_taken = datetime.now() - start
# logging.info(f'runtime was {time_taken}')

#rmse = feed_forward(x, W0, W1, W2, W3, b0, b1, b2, b3, y_true)
#print(f'ending rmse is {rmse}')
# final = pd.concat(df_all, sort=False)

# output final

# kfc = ('DE-NN', 'exploration') 
# out_dir = r'/home/wesley/repos/rbf-fd/data'
# output_name = '-'.join(kfc)
# output_loc = os.path.join(out_dir + os.sep + output_name + '.ods')

# logging.info(f'Saving to {output_loc}')
# with pd.ExcelWriter(output_loc ) as writer:
#     final.to_excel(writer, sheet_name = 'final', index=False)
    
if False:
    
    ycol = 'LMP'
    xcol = 'datetime'
    #x = lmp[xcol]
    x = test[xcol]
    y = test[ycol]
    f = NN_model.x
    #f = f[xcol]

    y_pred = DENN_forecast(f, W0, W1, W2, W3,
                 b0, b1, b2, b3)

    plt.figure(figsize=(12,6))
    plt.plot(x, y, label='Historical', linewidth=1)
    plt.plot(x, y_pred, color='r', label='DENN Predicted', linewidth=1)
    plt.xlabel('datetime')
    plt.ylabel('LMP')
    plt.legend(fontsize = 10, loc='upper left')
    plt.savefig(f'images/houston-lmp-con.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()

if False:

    # testing data

    # load weather data testing 2023
    # load LMP data training 2023

    lmp_ = load_testing_lmp()
    weather_ = load_testing_weather(x_cols)

    # merge load and weather

    test = pd.merge(weather_, lmp_, on = ['datetime'])
    X_ = test[x_cols].copy()

    ycol = 'LMP'
    y_pred_ = nn.predict(X_)
    test_rmse = np.sqrt(mean_squared_error(test[ycol], y_pred_))
    print(f'NN test RMSE {test_rmse}')

if False:

    m = len(lmp_)-1
    f = test[x_cols].copy()
    y_pred_ = DENN_forecast(f, W0, W1, W2, W3,
                    b0[:m,:], b1[:m,:], b2[:m,:], b3[:m,:])

    denn_rmse = np.sqrt(mean_squared_error(test[ycol], y_pred_))
    print(f'DENN test RMSE {denn_rmse}')