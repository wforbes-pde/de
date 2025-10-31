import pandas as pd
import itertools
import os
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
from pandas_ods_reader import read_ods
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from scipy.linalg import svd
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
import math
import seaborn as sn


class DEModelClass():
    
    def __init__(self, NP, g, F, CR, mutation_type, NPI, init, track_length,
                    F_refine, F_delta, lowerF, upperF,
                    mutation_refine, refine_param, 
                    CR_refine, CR_delta, lowerCR, upperCR,
                    fitness_metric, run_enh, seed):
        
        self.NP = NP
        self.g = g
        self.F = F
        self.CR = CR
        self.dir_path = r'/home/wesley/repos/data'

        self.mutation_type = mutation_type
        self.F_delta = F_delta
        self.NPI = NPI
        self.init = init
        self.lowerF = lowerF
        self.upperF = upperF
        self.track_length = track_length
        self.refine_param = refine_param
        self.F_refine = F_refine        
        self.mutation_refine = mutation_refine

        self.lowerCR = lowerCR
        self.upperCR = upperCR
        self.CR_refine = CR_refine
        self.CR_delta = CR_delta

        self.feed = feed_forward
             
        self.fitness_metric = fitness_metric
        self.run_enh = run_enh
        self.fitness = fitness 
        self.return_F_CR = return_F_CR
        self.return_mutation_list = return_mutation_list
        self.return_mutation_type = return_mutation_type
        self.mutation = mutation
        self.crossover_broadcast = crossover_broadcast
        self.return_running_avg_residual = return_running_avg_residual
        self.perform_svd_filter = perform_svd_filter
        self.perform_svd_scalar = perform_svd_scalar
        self.perform_svd_exp = perform_svd_exp
        self.perform_svd_log = perform_svd_log
        self.perform_clustering = perform_clustering
        self.perform_search = perform_search
        self.DENN_forecast = DENN_forecast
        self.plot = plot

    def set_run(self, run):
        self.run = run

    def set_seed(self, seed):
        self.seed = seed


class NNClass():
    
    def __init__(self, num_layers, n1, n2, n3, m):
        

        self.num_layers = num_layers
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        # feature dimension

        self.m = m

        # activation function

        self.activation_function = relu
    
    def set_ytrain_std(self, std_):
        self.ytrain_std = std_

    def set_phase(self, phase):
        self.phase = phase


class BuildingDataClass():
    
    def __init__(self, building, x_cols, target, load_path, weather_path, daytype):
        
        self.building = building
        self.x_cols = x_cols
        self.target = target
        self.daytype = daytype

        # load paths

        self.load = load_load(load_path)

        # training weather

        weather = pd.read_csv(weather_path)

        cols = ['valid'] + x_cols
        weather = weather[cols].copy()
        weather = weather.reset_index(drop=True)
        weather['tmpf'] = weather['tmpf'].astype(float)

        # missing values backfill

        weather = weather.fillna(method ='pad')

        # slice weather

        weather['valid'] = pd.to_datetime(weather['valid'])       
        weather.index = weather['valid']

        weather = weather.resample('h', on='valid').mean()
        weather = weather.reset_index(drop = False)

        # need to convert to hdd/cdd

        weather['datetime'] = pd.to_datetime(weather['valid'])
        del weather['valid']
        weather=weather.dropna()

        # training load

        load = pd.read_csv(load_path, parse_dates=['DateTime'])
        load['datetime'] = pd.to_datetime(load['DateTime'])
        del load['DateTime']

        lcols = ['datetime', 'RealPower']
        load = load[lcols].copy()
        load = load.rename(columns={'RealPower':'kWh'})
        load=load.reset_index(drop=True)
        load.index = load['datetime']

        load = load.resample('h', on='datetime').mean()
        load = load.reset_index(drop = False)
        load = load[load.kWh > 0].copy()

        # years 2015-2019

        # training: 2015-2017
        # validation: 2018
        # test: 2019

        training_year = [2015,2016,2017]
        validation_year = [2018]
        test_year = [2019]

        data = pd.merge(load, weather, on = ['datetime'])
        
        # day type or peak/off peak?
        
        if self.daytype is not None:
        
            data['DayOfWeek'] = data['datetime'].dt.weekday
            
            # The day of the week with Monday=0, Sunday=6.
            
            weekend_list = [5,6]
            weekend_mask = data['DayOfWeek'].isin(weekend_list)
            weekend = data[weekend_mask].copy()
            weekday = data[~weekend_mask].copy()
            
            if daytype == 'weekend':
                data = weekend.copy()
            if daytype == 'weekday':
                data = weekday.copy()

        data['Year'] = data['datetime'].dt.year

        # split data sets

        training_mask = data['Year'].isin(training_year)
        training_data = data[training_mask].copy()

        validation_mask = data['Year'].isin(validation_year)
        validation_data = data[validation_mask].copy()

        test_mask = data['Year'].isin(test_year)
        test_data = data[test_mask].copy()

        self.training = training_data
        self.validation = validation_data
        self.testing = test_data

        # train, validation, and testing

        self.X_train = self.training[self.x_cols]
        self.y_train = self.training[self.target]

        self.X_val = self.validation[self.x_cols]
        self.y_val = self.validation[self.target]

        self.X_test = self.testing[self.x_cols]
        self.y_test = self.testing[self.target]

    def set_application(self, application):
        self.application = application

    def set_exercise(self, exercise):
        self.exercise = exercise     


class ConcreteDataClass():
    
    def __init__(self, x_cols, target, concrete_path):
    
        self.x_cols = x_cols
        self.target = target

        # training load

        df = read_ods(concrete_path)

        # Separate features and target
        X = df.drop(columns=self.target)
        y = df[self.target]

        self.X = X
        self.y = y

        # test

        X_train_range = 733
        #X_val_range = 850
        X_test_range = 1100

        self.X_train = self.X[:X_train_range]
        #self.X_val = self.X[X_train_range:X_val_range]
        self.X_test = self.X[X_train_range:X_test_range]

        self.y_train = self.y[:X_train_range]
        #self.y_val = self.y[X_train_range:X_val_range]
        self.y_test = self.y[X_train_range:X_test_range]

        self.testing = self.y_test

    def set_application(self, application):
        self.application = application

    def set_exercise(self, exercise):
        self.exercise = exercise    
    

class MCMCClass():
    
    def __init__(self, mcmc_arg, burn_in, error_dist, error_std, 
                 run, pred_post_sample):
        
        run_mcmc, multiple_chain,chains = mcmc_arg
        self.run_mcmc = run_mcmc
        self.burn_in = burn_in
        self.error_dist = error_dist
        self.error_std = error_std
        self.run = run
        self.pred_post_sample = pred_post_sample
        self.multiple_chain = multiple_chain
        self.chains = chains
        self.run_mcmc_arg = mcmc_arg
        self.serial_chain_MCMC = serial_chain_MCMC
        self.multiple_chain_MCMC = multiple_chain_MCMC
        self.print = False

    def setup_mcmc_array(self, m,n1,n2,n3,NP, G):
        
        if self.multiple_chain:
            num_chains = self.chains
            num_samples = G-self.burn_in
            W0 = np.full((NP, num_samples, m, n1), 1.0, dtype=np.float32)
            W1 = np.full((NP, num_samples, n1, n2), 1.0, dtype=np.float32)
            W2 = np.full((NP, num_samples, n2, n3), 1.0, dtype=np.float32)
            W3 = np.full((NP, num_samples, n3, 1), 1.0, dtype=np.float32)

            b0 = np.full((NP, num_samples, 1, n1), 1.0, dtype=np.float32)
            b1 = np.full((NP, num_samples, 1, n2), 1.0, dtype=np.float32)
            b2 = np.full((NP, num_samples, 1, n3), 1.0, dtype=np.float32)
            b3 = np.full((NP, num_samples, 1, 1), 1.0, dtype=np.float32)
        
        if not self.multiple_chain:
            num_samples = G-self.burn_in
            W0 = np.full((num_samples, m, n1), 1.0, dtype=np.float32)
            W1 = np.full((num_samples, n1, n2), 1.0, dtype=np.float32)
            W2 = np.full((num_samples, n2, n3), 1.0, dtype=np.float32)
            W3 = np.full((num_samples, n3, 1), 1.0, dtype=np.float32)

            b0 = np.full((num_samples, 1, n1), 1.0, dtype=np.float32)
            b1 = np.full((num_samples, 1, n2), 1.0, dtype=np.float32)
            b2 = np.full((num_samples, 1, n3), 1.0, dtype=np.float32)
            b3 = np.full((num_samples, 1, 1), 1.0, dtype=np.float32)

        return W0, W1, W2, W3, b0, b1, b2, b3

    def set_acceptance_rate(self, rate):
        self.acceptance_rate = rate

    def set_top_chains(self, gen_fit_values, chains):
        if chains is not None:
            top_chains = np.argpartition(gen_fit_values, chains-1)[:chains]
            self.top_chains = list(top_chains)
        else:
            self.top_chains = []


def load_load(path):

    logging.info(f'loading load {path}')
    load = pd.read_csv(path, parse_dates=['DateTime'])
    load['datetime'] = pd.to_datetime(load['DateTime'])
    del load['DateTime']

    lcols = ['datetime', 'RealPower']
    load = load[lcols].copy()
    load = load.rename(columns={'RealPower':'kWh'})
    load=load.reset_index(drop=True)
    load.index = load['datetime']

    load = load.resample('h', on='datetime').mean()
    load = load.reset_index(drop = False)
    load = load.resample('D', on='datetime').sum()
    load = load.reset_index(drop = False)
    load = load[load.kWh > 0]

    return load


def fitness(x, W0, W1, W2, W3, b0, b1, b2, b3, y_, n_, fitness_metric, 
            NN_model):
    
    yp = feed_forward(x, W0, W1, W2, W3, b0, b1, b2, b3, n_, NN_model)

    # error function

    weights = None
    base_score = return_fitness_metric(y_, yp, fitness_metric, weights)    

    return base_score, yp


def feed_forward(x, W0, W1, W2, W3, b0, b1, b2, b3, n_, NN_model):

    if NN_model.num_layers == 3:
    
        s1 = x@W0 + b0
        #z1 = NN_model.activation_function(s1, NN_model.alpha)
        z1 = np.maximum(0,s1)

        s2 = z1@W1 + b1
        #z2 = NN_model.activation_function(s2, NN_model.alpha)
        z2 = np.maximum(0,s2)

        s3 = z2@W2 + b2
        #z3 = NN_model.activation_function(s3, NN_model.alpha)
        z3 = np.maximum(0,s3)

        #yp = NN_model.activation_function(z3@W3+b3, NN_model.alpha)
        yp = np.maximum(0,z3@W3+b3)

        #rmse_np = np.sqrt(np.mean((yp - y_train[None, :, :])**2, axis=(1, 2)))

    return yp


def return_fitness_metric(y_, yp, fitness_metric, weights):

    if fitness_metric == 'rmse':
        score = root_mean_squared_error(y_, yp, sample_weight=weights)

    if fitness_metric == 'mse':
        score = root_mean_squared_error(y_, yp, squared=True, sample_weight=weights)

    if fitness_metric == 'mae':
        score = mean_absolute_error(y_, yp, sample_weight=weights)

    if fitness_metric == 'mape':
        score = mean_absolute_percentage_error(y_, yp, sample_weight=weights)

    if fitness_metric == 'r2':
        score = 1-r2_score(y_, yp)

    return score


def relu(w):
    # np.maximum is much faster than masking
    w_ = np.maximum(0,w)
    return w_

def DENN_forecast(x, W0, W1, W2, W3,
                 b0, b1, b2, b3, NN_model, MCMC):
    
    # convert bias vectors into bias matrices
    
    n_ = len(x)

    if not MCMC.run_mcmc:

        b0 = np.repeat(b0, n_, axis=0)
        b1 = np.repeat(b1, n_, axis=0)
        b2 = np.repeat(b2, n_, axis=0)
        b3 = np.repeat(b3, n_, axis=0)

    s1 = x@W0 + b0
    z1 = NN_model.activation_function(s1)

    s2 = z1@W1 + b1
    z2 = NN_model.activation_function(s2)

    s3 = z2@W2 + b2
    z3 = NN_model.activation_function(s3)

    yp = NN_model.activation_function(z3@W3+b3)

    return yp


def plot_array(key, H):
     
    dir_path = '/home/wesley/repos/array'
    kfc = (key, ' Weight Std', )
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    plt.imshow(H, interpolation='none')
    #plt.close()
    plt.clf()

def plot_autocorrelation(key, p, q, test, parameter, run):

    #plt.figure(figsize=(18,6))
    lags_ = len(test)/4
    plot_acf(test, lags=lags_, markersize=1)
    
    #plt.xlabel('Lag')
    #plt.ylabel('AutoCorrelation')
    #plt.legend(fontsize = 10, loc='upper left')

    dir_path = '/home/wesley/repos/auto'
    kfc = (key, ' Chain Autocorrelation', str(p), str(q), ' Run', str(run))
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300)
    #plt.show()
    #plt.close('all')
    plt.clf()




def plot_weight_hist(key, p, q, test, parameter, run, fig, bins_):

    #plt.figure(figsize=(12,6))
    std = np.std(test)
    xmax = parameter + std
    xmin = parameter - std
    #plt.hist(test, bins='auto', range=[xmin,xmax])
    plt.hist(test, bins=bins_, histtype='stepfilled') # histtype='stepfilled',
    plt.axvline(parameter, color='k', linestyle='dashed', linewidth=1)
    
    plt.xlabel('Value')
    plt.ylabel('Count')
    #plt.legend(fontsize = 10, loc='upper left')

    dir_path = '/home/wesley/repos/hist'
    kfc = (key, ' Matrix Hist', str(p), str(q), ' Run', str(run), ' Bins', str(bins_))
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    #plt.show()
    #plt.close('all')
    plt.clf()

def plot_weight_trace_plot(key, p, q, test, run, chain_slice):

    x_ = np.arange(0,len(test))
    #plt.figure(figsize=(12,6))
    plt.plot(x_, test,  label='chain', linewidth=0.5)
    
    plt.xlabel('Index')
    plt.ylabel('Value')
    #plt.legend(fontsize = 10, loc='upper left')

    dir_path = '/home/wesley/repos/trace'
    kfc = (key, ' Trace Plot', str(p), str(q), 'ChainSlice', str(chain_slice), ' Run', str(run))
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    plt.clf()
    #plt.close('all')

def plot_CI(dataset, dtcol, samples_pred, application, daytype, run, mm):
    fig, ax = plt.subplots(figsize=(18,6)) # 18,6

    ax.plot(dataset[dtcol], dataset['pred_mean'], label='Mean', linewidth=1, color='black')
    ax.plot(dataset[dtcol], dataset['actual'], label='Actual', linewidth=1, color='red')
    ax.set_aspect('auto') 

    lower = np.percentile(samples_pred, 5, axis=0)
    upper = np.percentile(samples_pred, 95, axis=0)
    #lower=lower.reshape(len(lower),)
    #upper=upper.reshape(len(upper),)
    plt.fill_between(dataset['datetime'], dataset['lower'], dataset['upper'], alpha=0.5, label='5% and 95% confidence interval', color='cornflowerblue')
    plt.legend()
    plt.margins(x=0)
    plt.tight_layout()
    plt.xlabel('datetime')
    plt.ylabel('kWh')

    dir_path = '/home/wesley/repos/'
    kfc = ('Credible Interval', mm, application, str(daytype), 'Run', str(run))
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300)
    plt.clf()


def plot_CI_concrete(dataset, samples_pred, application, daytype, run, mm):
    fig, ax = plt.subplots(figsize=(18,6)) # 18,6

    ax.plot(dataset['index'], dataset['pred_mean'], label='Mean', linewidth=1, color='black')
    ax.plot(dataset['index'], dataset['actual'], label='Actual', linewidth=1, color='red')
    ax.set_aspect('auto') 

    plt.fill_between(dataset['index'], dataset['lower'], dataset['upper'], alpha=0.5, label='5% and 95% confidence interval', color='cornflowerblue')
    plt.legend()
    plt.margins(x=0)
    plt.tight_layout()
    plt.xlabel('index')
    plt.ylabel('Strength')

    dir_path = '/home/wesley/repos/'
    kfc = ('Credible Interval', mm, application, str(daytype), 'Run', str(run))
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300)
    plt.clf()



def random_uniform(gen_points, samples, NP_indices):

    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    low_ = -1e-1
    high_ = 1e-1

    # create random 3d arrays

    NP = len(NP_indices)
    total = NP*samples

    m,n = xgp_W0[0].shape
    X_W0 = np.random.uniform(low=low_, high=high_, size=(total,m,n))
    repeated_xgp_W0 = np.repeat(xgp_W0, repeats=samples, axis=0)
    rgp_W0 = repeated_xgp_W0.copy() + X_W0

    m,n = xgp_W1[0].shape
    X_W1 = np.random.uniform(low=low_, high=high_, size=(total,m,n))
    repeated_xgp_W1 = np.repeat(xgp_W1, repeats=samples, axis=0)
    rgp_W1 = repeated_xgp_W1.copy() + X_W1

    m,n = xgp_W2[0].shape
    X_W2 = np.random.uniform(low=low_, high=high_, size=(total,m,n))
    repeated_xgp_W2 = np.repeat(xgp_W2, repeats=samples, axis=0)
    rgp_W2 = repeated_xgp_W2.copy() + X_W2

    m,n = xgp_W3[0].shape
    X_W3 = np.random.uniform(low=low_, high=high_, size=(total,m,n))
    repeated_xgp_W3 = np.repeat(xgp_W3, repeats=samples, axis=0)
    rgp_W3 = repeated_xgp_W3.copy() + X_W3

    m,n = xgp_b0[0].shape
    X_b0 = np.random.uniform(low=low_, high=high_, size=(total,m,n))
    repeated_xgp_b0 = np.repeat(xgp_b0, repeats=samples, axis=0)
    rgp_b0 = repeated_xgp_b0.copy() + X_b0

    m,n = xgp_b1[0].shape
    X_b1 = np.random.uniform(low=low_, high=high_, size=(total,m,n))
    repeated_xgp_b1 = np.repeat(xgp_b1, repeats=samples, axis=0)
    rgp_b1 = repeated_xgp_b1.copy() + X_b1

    m,n = xgp_b2[0].shape
    X_b2 = np.random.uniform(low=low_, high=high_, size=(total,m,n))
    repeated_xgp_b2 = np.repeat(xgp_b2, repeats=samples, axis=0)
    rgp_b2 = repeated_xgp_b2.copy() + X_b2

    m,n = xgp_b3[0].shape
    X_b3 = np.random.uniform(low=low_, high=high_, size=(total,m,n))
    repeated_xgp_b3 = np.repeat(xgp_b3, repeats=samples, axis=0)
    rgp_b3 = repeated_xgp_b3.copy() + X_b3

    local = rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3

    return local


def svd_space(M, alpha):

    # M = weight matrix
    # alpha = percent of singular values to filter - unused!!
    # alpha = how many singular values to exclude

    U, S, V_T = svd(M)
    w0 = len(S)    
    Sigma = np.zeros((M.shape[0], M.shape[1]))

    S_ = np.diag(S)        
        
    #k = int(w0*alpha)
    S_[w0-alpha:] = 0
    Sigma[:w0, :w0] = S_
    M_ = U @ Sigma @ V_T
    return M_

def reconstruct_SVD(U, S, V_T):

    Sigma = np.zeros((U.shape[0], V_T.shape[1]))

    # populate Sigma with n x n diagonal matrix

    S_ = np.diag(S)
    w = len(S)
    Sigma[:w, :w] = S_
    M_ = U @ Sigma @ V_T
    return M_


def cluster_array(xgp, clustering_type, num_of_clusters):
    #logging.info(f'clustering type is {clustering_type}')
    # reshaping for sklearn
    # flatten each matrix
    
    #d = len(xgp.keys())
    d = len(xgp)
    a,b = xgp[0].shape
    c = a*b
    X = np.zeros((c,d))

    for j in np.arange(0,d):
    #for j in xgp.keys():
        X[:,j] = xgp[j].flatten()

    X = X.T

    # predetermined number of clusters for kmeans, spectral

    if clustering_type == 'kmeans':

        kmeans = KMeans(n_clusters=num_of_clusters, n_init=10) # random_state=42
        kmeans.fit(X)        
        c_kmeans = kmeans.predict(X)
        centers = kmeans.cluster_centers_
        clabels = c_kmeans

    # spectral clustering n_neighbors = num_of_clusters,

    if clustering_type == 'spectral':
        sc = SpectralClustering(n_clusters=num_of_clusters, n_init=10, affinity='nearest_neighbors', 
                                n_neighbors = d).fit(X)  # random_state=42

        # determine centers from clustering

        df = pd.DataFrame.from_dict({
                'id': list(sc.labels_) ,
                'data': list(X)
            })
        #centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).median().agg(np.array,1) original
        centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).mean().agg(np.array,1)
        centers = centers.reset_index(drop = True)
        #centers = np.array([np.broadcast_to(row, shape=(d)) for row in centers])
        centers = np.array([np.broadcast_to(row, shape=(a*b)) for row in centers])
        clabels = sc.labels_
    
    # agglomerative

    if clustering_type == 'agg':
        agg = AgglomerativeClustering(n_clusters=num_of_clusters)
        c_means = agg.fit_predict(X)
        clf = NearestCentroid()
        clf.fit(X, c_means)
        centers = clf.centroids_
        clabels = c_means


    # convert center points array into dict
    # convert center points array into 3d array
        
    centers = centers.T
    center_array = np.full((num_of_clusters, a, b), 1.0, dtype=np.float32)
        
    for j in np.arange(0,num_of_clusters):
        center_array[j] = centers[:,j].reshape(a,b)

    return center_array



def return_distribution_mode(Wp, W, key, run, multiple_chain, MCMC, model_name, top_chains):
    logging.info(f'starting MCMC {key} run {run}')

    # handle single or multiple chains

    if not multiple_chain:
        M, df_list, T = return_single_chain_estimate(Wp, W, run, key, MCMC, model_name)

    if multiple_chain:
        M, df_list, T = return_multiple_chain_estimate(Wp, W, run, key, MCMC, model_name, top_chains)
    
    plt.close('all')
    return M, df_list, T

def clean_chain(W):

    # Identify slices (along axis 0) that contain 1s
    mask = np.any(W == 1, axis=(1, 2))  # True where a slice has at least one 1
    W_ = W[~mask].copy()  # Keep only slices without 1s

    mask = np.any(W_ == 0, axis=(1, 2))
    result = W_[~mask].copy()
    return result

def gelman_rubin(mcmc_chain, gen_points, run, ):

    gelman = []

    W0, W1, W2, W3, b0, b1, b2, b3 = mcmc_chain
    gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3 = gen_points

    W0_df = determine_chain_mean_variance(gb_W0, W0, 'W0', run)
    W1_df = determine_chain_mean_variance(gb_W1, W1, 'W1', run)
    W2_df = determine_chain_mean_variance(gb_W2, W2, 'W2', run)
    W3_df = determine_chain_mean_variance(gb_W3, W3, 'W3', run)

    b0_df = determine_chain_mean_variance(gb_b0, b0, 'b0', run)
    b1_df = determine_chain_mean_variance(gb_b1, b1, 'b1', run)
    b2_df = determine_chain_mean_variance(gb_b2, b2, 'b2', run)
    b3_df = determine_chain_mean_variance(gb_b3, b3, 'b3', run)

    gelman.append(W0_df)
    gelman.append(W1_df)
    gelman.append(W2_df)
    gelman.append(W3_df)

    gelman.append(b0_df)
    gelman.append(b1_df)
    gelman.append(b2_df)
    gelman.append(b3_df)

    diag = pd.concat(gelman)

    L = len(W0[0])
    J = len(W0)

    # Mean of the means of all chains (grand mean)

    key_ = ['key', 'row', 'column']
    key_cols_ = ['chain_mean']
    grand_mean = diag.groupby(key_)[key_cols_].aggregate(['mean'])
    grand_mean = grand_mean.reset_index(drop=False)
    grand_mean = grand_mean.droplevel(1,axis=1)
    grand_mean['grand_mean'] = grand_mean['chain_mean']

    # between chain variance

    key_ = ['key', 'row', 'column']
    #var_of_chain_means = diag.groupby(key_)[key_cols_].aggregate(np.var)
    var_of_chain_means = diag.groupby(key_)[key_cols_].aggregate('var')
    var_of_chain_means = var_of_chain_means.reset_index(drop=False)
    var_of_chain_means['B'] = var_of_chain_means['chain_mean']*(L*J/(J-1))
    del var_of_chain_means['chain_mean']

    # average of within chain variance

    key_ = ['key', 'row', 'column']
    key_cols_ = ['chain_variance']
    mean_of_chain_var = diag.groupby(key_)[key_cols_].aggregate(['mean'])
    mean_of_chain_var = mean_of_chain_var.reset_index(drop=False)
    mean_of_chain_var = mean_of_chain_var.droplevel(1,axis=1)
    mean_of_chain_var['W'] = mean_of_chain_var['chain_variance'] 
    del mean_of_chain_var['chain_variance']

    # statistic

    # length of chain

    # how many chains

    df = pd.merge(grand_mean, var_of_chain_means, on = key_ )
    df = pd.merge(df, mean_of_chain_var, on = key_)

    df['R'] = ( (L-1)/L*df['W'] + 1/L*df['B'] ) / df['W']

    return diag, df
    

def determine_chain_mean_variance(Wp, W, key, run):
    #logging.info(f'starting Gelman-Rubin {key} run {run}')
    df_list = []

    # convert markov chain to array

    a,b = Wp[0].shape # shape of weight matrix
    M = np.zeros((a,b))
    S = np.zeros((a,b))

    # number of chains

    c = len(W)

    # length of a particular chain

    d = len(W[0])

    # 3d array

    # loop through each chain
    # W_ is current chain that should be d long

    for t in np.arange(0,c):
        W_ = W[t]
        T = np.zeros((d,a,b))

        # broadcast entire long Markov chain "matrix" into 3d array

        try:
            for k in np.arange(0,d):
                #logging.info(f'starting Gelman-Rubin {key} run {run} k value {k}')
                T[k,:,:] = W_[k]
        except:
            logging.info(f'starting Gelman-Rubin {key} run {run} k value {k} {W_}')

        # construct markov chain array for each i,j element in the weight matrices/bias vectors

        for p in np.arange(0,a):
            for q in np.arange(0,b):
                test = T[:,p,q]
                #plot_weight_trace_plot(key, p, q, test, run, 'testing')
                # Mean value of chain j
                # variance of chain j

                current_chain_mean = np.mean(test)
                current_chain_var = np.var(test)

                df = pd.DataFrame({'key':[key], 'row':[p], 'column':[q], 'chain_mean':[current_chain_mean], 
                                'chain_variance':[current_chain_var], 'run':[run], 'chain':[t],  'chain':[t], })
                df_list.append(df)

    df_all = pd.concat(df_list, sort=False)

    return df_all

def return_single_chain_estimate(Wp, W, run, key, MCMC, model_name):

    a,b = Wp[0].shape # shape of weight matrix
    M = np.zeros((a,b))

    # length of chain

    if MCMC.pred_post_sample not in ['default']:
        total_sample_length = MCMC.pred_post_sample
    else:
        total_sample_length = len(W)

    # 3d array W=T

    T = W
    T = clean_chain(T)
    
    plt.figure(1,figsize=(12,6),) # weight kde plot
    plt.figure(2, figsize=(12,6)) # trace plot
    plt.figure(3, figsize=(12,6)) # autocorr
    plt.figure(4, figsize=(12,6)) # autocorr
    plt.suptitle(f"{key}", fontsize = 14)
    c=1
    d=1
    e=0

    # create the figure and axes
    #fig, axes = plt.subplots(a, b, sharex=True, sharey=True, squeeze=False, figsize=(12,6), num=3)
    fig, axes = plt.subplots(5, 1, sharex=True, sharey=True, squeeze=False, figsize=(12,6), num=3)
    #print(f'a {a} b {b}')
    for p in np.arange(0,a):
        for q in np.arange(0,b):
            chain_vector = T[-total_sample_length:,p,q]

            # statistical mode

            mode1 = stats.mode(chain_vector,keepdims=True,axis=None).mode[0]
            M[p,q] = mode1

            # plot weight histogram
            
            bw_adjust = 1
            
            if MCMC.print and False:
                plt.figure(1)
                plt.subplot(a, b, c)
                sn.kdeplot(data=pd.DataFrame(chain_vector), color="green", fill=True, label=str(p) + str(q), bw_adjust=bw_adjust)
                plt.legend(fontsize = 10, loc='upper right')
                #plot_kde(key, p, q, test, mode1, run, bw_adjust)
            #     plot_weight_hist(key, p, q, test, mode1, run, fig,w)
                c=c+1

            # plot trace plot
                
            if MCMC.print and p==q and p <= 15 and key in ['W0', 'W1', 'W2']:
                plt.figure(2)
                plt.subplot(4, 4, p+1)
                x_ = np.arange(0,len(chain_vector))
                plt.plot(x_, chain_vector, label=str(p) + str(q), linewidth=0.5)
                plt.legend(fontsize = 10, loc='upper right')
                plt.xlabel('Index')
                plt.ylabel('Value')
                #plot_weight_trace_plot(key, p, q, test, run, chain_slice)
                d=d+1

            if MCMC.print and p <= 5 and key not in ['W0', 'W1', 'W2']:
                plt.figure(2)
                plt.subplot(4, 4, p+1)
                x_ = np.arange(0,len(chain_vector))
                plt.plot(x_, chain_vector, label=str(p) + str(q), linewidth=0.5)
                plt.legend(fontsize = 10, loc='upper right')
                plt.xlabel('Index')
                plt.ylabel('Value')
                #plot_weight_trace_plot(key, p, q, test, run, chain_slice)
                d=d+1

            # autocorrelation

            if MCMC.print and p==q and p <= 4 and key in ['W0', 'W1', 'W2'] and False:
                figs = plt.figure(3)
                lags_ = len(chain_vector)/4
                #plot_acf(chain_vector, ax=axes[p][q],lags=lags_, markersize=1)
                plot_acf(chain_vector, ax=axes[p][0],lags=lags_, markersize=1)
                #plt.legend(fontsize = 10, loc='upper right') not used
                label_=str(p) + str(q)
                #axes[p][q].legend([label_] ,loc='upper right', fontsize = 'x-small')
                axes[p][0].legend([label_] ,loc='upper right', fontsize = 'x-small')
                e=e+1
    
    if MCMC.print:
    
        # kde plot

        # plt.figure(1)
        # plt.suptitle(f"{key}", fontsize = 14)
        # kfc = (model_name, key, 'Matrix KDE TEST', str(p), str(q), 'Run', str(run))
        # output_name = ' '.join(kfc)
        # dir_path = '/home/wesley/repos/hist'
        # output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
        # logging.info(f'Saving to {output_loc}')
        # plt.tight_layout()
        # plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')   
        # plt.clf()

        # trace plot

        plt.figure(2)
        plt.suptitle(f"{key}", fontsize = 14)
        dir_path = '/home/wesley/repos/trace'
        kfc = (model_name, key, 'Trace Plot', str(p), str(q), 'Run', str(run))
        output_name = ' '.join(kfc)
        output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
        logging.info(f'Saving to {output_loc}')
        plt.tight_layout()
        plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
        plt.clf()

        # autocorrelation plot

        plt.figure(3)
        dir_path = '/home/wesley/repos/auto'
        kfc = (model_name, key, 'Chain Autocorrelation', str(p), str(q), 'Run', str(run))
        output_name = ' '.join(kfc)
        output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
        logging.info(f'Saving to {output_loc}')
        plt.tight_layout()
        plt.savefig(output_loc, dpi=300)
        plt.clf()
        
        # fitter plot

        # plt.figure(4)
        # dir_path = '/home/wesley/repos/fitter'
        # kfc = (model_name, key, ' Fitter', str(p), str(q), ' Run', str(run))
        # output_name = ' '.join(kfc)
        # output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
        # logging.info(f'Saving to {output_loc}')
        # plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
        # plt.clf()

    derta = None
    return M, derta, T



def return_multiple_chain_estimate(Wp, W, run, key, MCMC, model_name,top_chains):
    #logging.info(f'parallel chain {key} run {run}')
    a,b = Wp[0].shape # shape of weight matrix
    M = np.zeros((a,b))

    # length of chain      
    # sometimes chain is unexpectedly short one value - TODO?

    # TSTING

    slices = []

    for i in range(MCMC.chains):
        slices.append(W[i])
    chain_sample = np.concatenate(slices, axis=0)
    # 3d array
    
    # loop through each chain to create master 3d array
    # init top chain
    
    chain_sample = clean_chain(chain_sample)

    plt.figure(1, figsize=(12,6))
    plt.figure(3, figsize=(12,6))
    c=1
    e=1

    for p in np.arange(0,a):
        for q in np.arange(0,b):

            chain_vector = chain_sample[:,p,q]

            #plot_weight_trace_plot(key, p, q, test, run, chain_slice)
            # statistical mode

            mode1 = stats.mode(chain_vector,keepdims=True,axis=None).mode[0]
            M[p,q] = mode1

            # plot weight histogram

            bw_adjust = 1
            
            if MCMC.print:
                plt.figure(1)
                plt.subplot(a, b, c)
                #plot_kde(key, p, q, test, mode1, run, bw_adjust)
                sn.kdeplot(data=pd.DataFrame(chain_vector), color="green", fill=True, label=str(p) + str(q), bw_adjust=bw_adjust)
                plt.legend(fontsize = 10, loc='upper right')
                #plot_weight_hist(key, p, q, test, mode1, run, fig,w)
                c=c+1

            # plot trace plot
                
            if MCMC.print and False:
                #plot_weight_trace_plot(key, p, q, test, run, chain_slice)
                d=d+1

            # autocorrelation

            if MCMC.print and False:                
                #plot_autocorrelation(key, p, q, test, mode1, run)
                d=d+1

    if MCMC.print:
    
        plt.figure(1)
        plt.suptitle(f"{key}", fontsize = 14)
        kfc = (model_name, key, 'Matrix KDE TEST', str(p), str(q), 'Run', str(run))
        output_name = ' '.join(kfc)
        dir_path = '/home/wesley/repos/hist'
        output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
        logging.info(f'Saving to {output_loc}')
        plt.tight_layout()
        plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
        #plt.clf()

        # plt.figure(2)
        # plt.suptitle(f"{key}", fontsize = 14)
        # dir_path = '/home/wesley/repos/trace'
        # kfc = (key, ' Trace Plot TEST', str(p), str(q), 'ChainSlice', str(chain_slice), ' Run', str(run))
        # output_name = ' '.join(kfc)
        # output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
        # logging.info(f'Saving to {output_loc}')
        # plt.tight_layout()
        # #fig2.layout.update(matches='x')
        # plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
        # plt.clf()

        # plt.figure(3)
        # kfc = (key, ' Matrix AUTO TEST', str(p), str(q), ' Run', str(run))
        # output_name = ' '.join(kfc)
        # dir_path = '/home/wesley/repos/hist'
        # output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
        # logging.info(f'Saving to {output_loc}')
        # plt.tight_layout()
        # plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
        #plt.clf()

        # fitter plot

        plt.figure(3)
        plt.suptitle(f"{key}", fontsize = 14)
        dir_path = '/home/wesley/repos/fitter'
        kfc = (key, ' Fitter', str(p), str(q), ' Run', str(run))
        output_name = ' '.join(kfc)
        output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
        logging.info(f'Saving to {output_loc}')
        plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')

    #derta = pd.concat(df_list, sort=False)
    derta = None
    return M, derta, chain_sample


def plot_kde(key, p, q, x, mode1, run, w):

    #sn.kdeplot(data=pd.DataFrame(x), color="green",fill=True, label=key + str(p) + str(q), bw_adjust=w)
    sn.kdeplot(data=pd.DataFrame(x), color="green", fill=True, label=str(p) + str(q), bw_adjust=w)
    
    plt.legend(fontsize = 10, loc='upper right')
    dir_path = '/home/wesley/repos/hist'
    kfc = (key, ' Matrix KDE', str(p), str(q), ' Run', str(run), str(w))
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    #plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    #plt.clf()



def return_F_CR(flag, lowerF, upperF, F_delta, F_, DE_model, NN_model):

    # ways in which F_CR can vary
    # across generation
    # across index
    # across weight/bias
    # across component

    if flag == 'default':
        F_W0 = F_
        F_W1 = F_
        F_W2 = F_
        F_W3 = F_

        F_b0 = F_
        F_b1 = F_
        F_b2 = F_
        F_b3 = F_

    if flag == 'variable':
        movie_list = np.arange(lowerF,upperF,F_delta)
        movie_list = np.round(movie_list,2)
        movie_list = list(movie_list)

        F_W0 = random.choice(movie_list)
        F_W1 = F_W0
        F_W2 = F_W0
        F_W3 = F_W0

        F_b0 = F_W0
        F_b1 = F_W0
        F_b2 = F_W0
        F_b3 = F_W0
    
    if flag == 'weight_variable':
        movie_list = np.arange(lowerF,upperF,F_delta)
        movie_list = np.round(movie_list,2)
        movie_list = list(movie_list)

        F_W0 = random.choice(movie_list)
        F_W1 = random.choice(movie_list)
        F_W2 = random.choice(movie_list)
        F_W3 = random.choice(movie_list)

        F_b0 = random.choice(movie_list)
        F_b1 = random.choice(movie_list)
        F_b2 = random.choice(movie_list)
        F_b3 = random.choice(movie_list)

    if flag == 'weight_variable_index':

        # Create a 3D array of ones with shape (NP, m, n)
        F_W0_base_array = np.ones((NP, NN_model.m, NN_model.n1))
        random_numbers = np.random.uniform(low=lowerF, high=upperF, size=NP)
        random_numbers_reshaped = random_numbers.reshape(NP, 1, 1)
        F_W0 = F_W0_base_array * random_numbers_reshaped

        F_W1_base_array = np.ones((NP, NN_model.n1, NN_model.n2))
        random_numbers = np.random.uniform(low=lowerF, high=upperF, size=NP)
        random_numbers_reshaped = random_numbers.reshape(NP, 1, 1)
        F_W1 = F_W1_base_array * random_numbers_reshaped

        F_W2_base_array = np.ones((NP, NN_model.n2, NN_model.n3))
        random_numbers = np.random.uniform(low=lowerF, high=upperF, size=NP)
        random_numbers_reshaped = random_numbers.reshape(NP, 1, 1)
        F_W2 = F_W2_base_array * random_numbers_reshaped

        F_W3_base_array = np.ones((NP, NN_model.n3, 1))
        random_numbers = np.random.uniform(low=lowerF, high=upperF, size=NP)
        random_numbers_reshaped = random_numbers.reshape(NP, 1, 1)
        F_W3 = F_W3_base_array * random_numbers_reshaped

        F_b0_base_array = np.ones((NP, 1, NN_model.n1))
        random_numbers = np.random.uniform(low=lowerF, high=upperF, size=NP)
        random_numbers_reshaped = random_numbers.reshape(NP, 1, 1)
        F_b0 = F_b0_base_array * random_numbers_reshaped

        F_b1_base_array = np.ones((NP, 1, NN_model.n2))
        random_numbers = np.random.uniform(low=lowerF, high=upperF, size=NP)
        random_numbers_reshaped = random_numbers.reshape(NP, 1, 1)
        F_b1 = F_b1_base_array * random_numbers_reshaped

        F_b2_base_array = np.ones((NP, 1, NN_model.n3))
        random_numbers = np.random.uniform(low=lowerF, high=upperF, size=NP)
        random_numbers_reshaped = random_numbers.reshape(NP, 1, 1)
        F_b2 = F_b2_base_array * random_numbers_reshaped

        F_b3_base_array = np.ones((NP, 1, 1))
        random_numbers = np.random.uniform(low=lowerF, high=upperF, size=NP)
        random_numbers_reshaped = random_numbers.reshape(NP, 1, 1)
        F_b3 = F_b3_base_array * random_numbers_reshaped

    if flag == 'general': # varies index and dimension
        NP = DE_model.NP
        F_W0 = np.random.uniform(low=lowerF, high=upperF, size=(NP,NN_model.m,NN_model.n1))
        F_W1 = np.random.uniform(low=lowerF, high=upperF, size=(NP,NN_model.n1,NN_model.n2))
        F_W2 = np.random.uniform(low=lowerF, high=upperF, size=(NP,NN_model.n2,NN_model.n3))
        F_W3 = np.random.uniform(low=lowerF, high=upperF, size=(NP,NN_model.n3,1))

        F_b0 = np.random.uniform(low=lowerF, high=upperF, size=(NP,1,NN_model.n1))
        F_b1 = np.random.uniform(low=lowerF, high=upperF, size=(NP,1,NN_model.n2))
        F_b2 = np.random.uniform(low=lowerF, high=upperF, size=(NP,1,NN_model.n3))
        F_b3 = np.random.uniform(low=lowerF, high=upperF, size=(NP,1,1))

    return F_W0, F_W1, F_W2, F_W3, F_b0, F_b1, F_b2, F_b3

def return_mutation_type(flag, mutation_list, mutation_default):

    if flag == 'default':
        mutation_W0 = mutation_default
        mutation_W1 = mutation_default
        mutation_W2 = mutation_default
        mutation_W3 = mutation_default

        mutation_b0 = mutation_default
        mutation_b1 = mutation_default
        mutation_b2 = mutation_default
        mutation_b3 = mutation_default

    if flag == 'variable':
        mutation_W0 = random.choice(mutation_list)
        mutation_W1 = mutation_W0
        mutation_W2 = mutation_W0
        mutation_W3 = mutation_W0

        mutation_b0 = mutation_W0
        mutation_b1 = mutation_W0
        mutation_b2 = mutation_W0
        mutation_b3 = mutation_W0

    if flag == 'weight_variable':
        mutation_W0 = random.choice(mutation_list)
        mutation_W1 = random.choice(mutation_list)
        mutation_W2 = random.choice(mutation_list)
        mutation_W3 = random.choice(mutation_list)

        mutation_b0 = random.choice(mutation_list)
        mutation_b1 = random.choice(mutation_list)
        mutation_b2 = random.choice(mutation_list)
        mutation_b3 = random.choice(mutation_list)
        
    mutation_op = mutation_W0, mutation_W1, mutation_W2, mutation_W3, mutation_b0, mutation_b1, mutation_b2, mutation_b3
    
    return mutation_op


def return_mutation_list(NP):

    if NP >= 4:
        mutation_list = ['best', 'random']

    if NP >= 6:
        mutation_list = ['best', 'best2', 'random', 'random2',]

    if NP >= 8:
        mutation_list = ['best', 'best2', 'best3', 'random', 'random2', 'random3']

    return mutation_list


def plot(x_train, target, y_t, y_p, label, ext):
    plt.figure(figsize=(16,6))
    plt.plot(x_train, y_t, label='Historical', linewidth=1)
    plt.plot(x_train, y_p, color='m', label=label, linewidth=1)
    plt.xlabel('datetime')
    plt.ylabel(target)
    if target=='LMP':
        plt.ylim(-20,200)
    plt.legend(fontsize = 10, loc='upper left')
    plt.savefig(f'/home/wesley/repos/images/{ext}.png',
                format='png',
                dpi=300,
                bbox_inches='tight')    
    # plt.savefig(f'/home/wesley/repos/images/{label}.png',
    #             format='png',
    #             dpi=300,
    #             bbox_inches='tight')
    #plt.show()
    plt.close()
    #output_loc = f'/home/wesley/repos/images/houston-lmp-denn-v-test-{season}-{run}-{aindex}.csv'
    #y_2023_.to_csv(output_loc)


def serial_chain_MCMC(gen, xfit, zfit, mindex, run_mcmc, ratio, burn_in,j,
                    MCMC, alpha, x_points, z_points, 
                    W0, W1, W2, W3, b0, b1, b2, b3):
    
    # shift

    mgen = gen-burn_in
    
    x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3 = x_points
    z_W0, z_W1, z_W2, z_W3, z_b0, z_b1, z_b2, z_b3 = z_points

    MCMC.set_acceptance_rate(0)

    # when proposed candidate improves fitness ratio = 1; otherwise ratio < 1

    if ratio == 1:
        MCMC.set_acceptance_rate(1)
        W0[mgen,:,:] = z_W0[j]
        W1[mgen,:,:] = z_W1[j]
        W2[mgen,:,:] = z_W2[j]
        W3[mgen,:,:] = z_W3[j]

        b0[mgen,:,:] = z_b0[j]
        b1[mgen,:,:] = z_b1[j]
        b2[mgen,:,:] = z_b2[j]
        b3[mgen,:,:] = z_b3[j]
    
    if ratio < 1 and alpha <= ratio:
        MCMC.set_acceptance_rate(1)
        if len(W0) == 0:
            W0[mgen,:,:] = x_W0[j]
            W1[mgen,:,:] = x_W1[j]
            W2[mgen,:,:] = x_W2[j]
            W3[mgen,:,:] = x_W3[j]

            b0[mgen,:,:] = x_b0[j]
            b1[mgen,:,:] = x_b1[j]
            b2[mgen,:,:] = x_b2[j]
            b3[mgen,:,:] = x_b3[j]  
        
        if len(W0) > 0:
            W0[mgen,:,:] = z_W0[j]
            W1[mgen,:,:] = z_W1[j]
            W2[mgen,:,:] = z_W2[j]
            W3[mgen,:,:] = z_W3[j]

            b0[mgen,:,:] = z_b0[j]
            b1[mgen,:,:] = z_b1[j]
            b2[mgen,:,:] = z_b2[j]
            b3[mgen,:,:] = z_b3[j]

    if run_mcmc and ratio < 1 and alpha > ratio:
        MCMC.set_acceptance_rate(0)
        if len(W0) == 0:
            W0[mgen,:,:] = x_W0[j]
            W1[mgen,:,:] = x_W1[j]
            W2[mgen,:,:] = x_W2[j]
            W3[mgen,:,:] = x_W3[j]

            b0[mgen,:,:] = x_b0[j]
            b1[mgen,:,:] = x_b1[j]
            b2[mgen,:,:] = x_b2[j]
            b3[mgen,:,:] = x_b3[j]
        
        if len(W0) > 0:
            W0[mgen,:,:] = W0[mgen-1,:,:]
            W1[mgen,:,:] = W1[mgen-1,:,:]
            W2[mgen,:,:] = W2[mgen-1,:,:]
            W3[mgen,:,:] = W3[mgen-1,:,:]

            b0[mgen,:,:] = b0[mgen-1,:,:]
            b1[mgen,:,:] = b1[mgen-1,:,:]
            b2[mgen,:,:] = b2[mgen-1,:,:]
            b3[mgen,:,:] = b3[mgen-1,:,:]
    
    return W0, W1, W2, W3, b0, b1, b2, b3


def multiple_chain_MCMC(gen, xfit, zfit, mindex, run_mcmc, ratio, burn_in, pop_index,
                    MCMC, alpha, x_points, z_points, 
                    W0, W1, W2, W3, b0, b1, b2, b3):    
        
    # shift

    mgen = gen-burn_in
    j = MCMC.top_chains.index(pop_index)
    
    x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3 = x_points
    z_W0, z_W1, z_W2, z_W3, z_b0, z_b1, z_b2, z_b3 = z_points

    MCMC.set_acceptance_rate(0)

    # when proposed candidate improves fitness ratio = 1; otherwise ratio < 1

    # shift

    mgen = gen-burn_in
    
    x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3 = x_points
    z_W0, z_W1, z_W2, z_W3, z_b0, z_b1, z_b2, z_b3 = z_points

    MCMC.set_acceptance_rate(0)

    # when proposed candidate improves fitness ratio = 1; otherwise ratio < 1

    if ratio == 1:
        MCMC.set_acceptance_rate(1)
        W0[j,mgen,:,:] = z_W0[pop_index]
        W1[j,mgen,:,:] = z_W1[pop_index]
        W2[j,mgen,:,:] = z_W2[pop_index]
        W3[j,mgen,:,:] = z_W3[pop_index]

        b0[j,mgen,:,:] = z_b0[pop_index]
        b1[j,mgen,:,:] = z_b1[pop_index]
        b2[j,mgen,:,:] = z_b2[pop_index]
        b3[j,mgen,:,:] = z_b3[pop_index]
    
    if ratio < 1 and alpha <= ratio:
        MCMC.set_acceptance_rate(1)
        if len(W0) == 0:
            W0[j,mgen,:,:] = x_W0[pop_index]
            W1[j,mgen,:,:] = x_W1[pop_index]
            W2[j,mgen,:,:] = x_W2[pop_index]
            W3[j,mgen,:,:] = x_W3[pop_index]

            b0[j,mgen,:,:] = x_b0[pop_index]
            b1[j,mgen,:,:] = x_b1[pop_index]
            b2[j,mgen,:,:] = x_b2[pop_index]
            b3[j,mgen,:,:] = x_b3[pop_index]
        
        if len(W0) > 0:
            W0[j,mgen,:,:] = z_W0[pop_index]
            W1[j,mgen,:,:] = z_W1[pop_index]
            W2[j,mgen,:,:] = z_W2[pop_index]
            W3[j,mgen,:,:] = z_W3[pop_index]

            b0[j,mgen,:,:] = z_b0[pop_index]
            b1[j,mgen,:,:] = z_b1[pop_index]
            b2[j,mgen,:,:] = z_b2[pop_index]
            b3[j,mgen,:,:] = z_b3[pop_index]

    if run_mcmc and ratio < 1 and alpha > ratio:
        MCMC.set_acceptance_rate(0)
        if len(W0) == 0:
            W0[j,mgen,:,:] = x_W0[pop_index]
            W1[j,mgen,:,:] = x_W1[pop_index]
            W2[j,mgen,:,:] = x_W2[pop_index]
            W3[j,mgen,:,:] = x_W3[pop_index]

            b0[j,mgen,:,:] = x_b0[pop_index]
            b1[j,mgen,:,:] = x_b1[pop_index]
            b2[j,mgen,:,:] = x_b2[pop_index]
            b3[j,mgen,:,:] = x_b3[pop_index]
        
        if len(W0) > 0:
            W0[j,mgen,:,:] = W0[j,mgen-1,:,:]
            W1[j,mgen,:,:] = W1[j,mgen-1,:,:]
            W2[j,mgen,:,:] = W2[j,mgen-1,:,:]
            W3[j,mgen,:,:] = W3[j,mgen-1,:,:]

            b0[j,mgen,:,:] = b0[j,mgen-1,:,:]
            b1[j,mgen,:,:] = b1[j,mgen-1,:,:]
            b2[j,mgen,:,:] = b2[j,mgen-1,:,:]
            b3[j,mgen,:,:] = b3[j,mgen-1,:,:]
    
    return W0, W1, W2, W3, b0, b1, b2, b3

def return_running_avg_residual(i, value, gen_fitness_list, resid_tracking_list, track_len):
    gen_fitness_list.append(value)
    gen_train_residual = gen_fitness_list[i]-gen_fitness_list[i-1]
    resid_tracking_list.append(gen_train_residual)
    running_avg_residual = sum(resid_tracking_list[-track_len:])/len(resid_tracking_list[-track_len:])
    return gen_fitness_list, gen_train_residual, resid_tracking_list, running_avg_residual


def create_distinct_indices(NP_indices, test, NP, indices, num_selected):
    # Ensure inputs are NumPy arrays
    NP_indices = np.array(NP_indices)  # Shape: (NP,), e.g., (10,)
    indices = np.array(indices)        # Shape: (NP,), e.g., (10,)
    
    # Initialize test as an empty array
    test = np.zeros((len(NP_indices), num_selected), dtype=int)  # Shape: (NP, 3), e.g., (10, 3)
    
    # Generate distinct indices for each NP_index
    for idx, j in enumerate(NP_indices):
        # Create a mask to exclude the current NP_index
        valid_indices = indices[indices != j]  # Exclude j from indices
        # Randomly select num_selected indices without replacement
        selected = np.random.choice(valid_indices, size=num_selected, replace=False)
        test[idx] = selected
    
    return test.tolist()  # Return as list to match original function's output



def return_F(key, F_one, F_two, F_three):

    F_W0, F_W1, F_W2, F_W3, F_b0, F_b1, F_b2, F_b3 = F_one
    F2_W0, F2_W1, F2_W2, F2_W3, F2_b0, F2_b1, F2_b2, F2_b3 = F_two
    F3_W0, F3_W1, F3_W2, F3_W3, F3_b0, F3_b1, F3_b2, F3_b3 = F_three

    if key == 'W0':
        F_1 = F_W0
        F_2 = F2_W0
        F_3 = F3_W0

    if key == 'W1':
        F_1 = F_W1
        F_2 = F2_W1
        F_3 = F3_W1

    if key == 'W2':
        F_1 = F_W2
        F_2 = F2_W2
        F_3 = F3_W2

    if key == 'W3':
        F_1 = F_W3
        F_2 = F2_W3
        F_3 = F3_W3

    if key == 'b0':
        F_1 = F_b0
        F_2 = F2_b0
        F_3 = F3_b0

    if key == 'b1':
        F_1 = F_b1
        F_2 = F2_b1
        F_3 = F3_b1

    if key == 'b2':
        F_1 = F_b2
        F_2 = F2_b2
        F_3 = F3_b2

    if key == 'b3':
        F_1 = F_b3
        F_2 = F2_b3
        F_3 = F3_b3
    
    return F_1, F_2, F_3

def mutation(NP, NP_indices, F_one, F_two, F_three, x_weight, MCMC, gen_best_x_weight, mutation_type, key):
    F_1, F_2, F_3 = return_F(key, F_one, F_two, F_three)
    y = mutate_(NP, NP_indices, F_1, F_2, F_3, x_weight, MCMC, mutation_type, gen_best_x_weight)
    return y


def determine_num_indices(mutation_type):
        
    if mutation_type == 'random':
        num_selected = 3

    if mutation_type == 'random2':
        num_selected = 5

    if mutation_type == 'random3':
        num_selected = 7    

    if mutation_type == 'best':
        num_selected = 2

    if mutation_type == 'best2':
        num_selected = 4

    if mutation_type == 'best3':
        num_selected = 6

    return num_selected


def mutate_logic(test, F, F2, F3, x, mutation_type, gen_best, noise,
                      NP, NP_indices):
    
    s1,s2,s3=noise.shape

    if mutation_type in ['random']:
        i = test[:, 0]  # Shape: (10,)
        j = test[:, 1]  # Shape: (10,)
        k = test[:, 2]  # Shape: (10,)
        v1 = x[j]       # Shape: (10, 7, 35)
        v2 = x[k]       # Shape: (10, 7, 35)

    if mutation_type in ['best']:
        j = test[:, 0]  # Shape: (10,)
        k = test[:, 1]  # Shape: (10,)
        v1 = x[j]
        v2 = x[k]    
    
    if mutation_type in ['random2']:
        i = test[:, 0]  # Shape: (10,)
        j = test[:, 1]  # Shape: (10,)
        k = test[:, 2]  # Shape: (10,)
        l = test[:, 3]
        m = test[:, 4]
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]
        
    if mutation_type in ['best2']:
        j = test[:, 0]  # Shape: (10,)
        k = test[:, 1]  # Shape: (10,)
        l = test[:, 2]
        m = test[:, 3]
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]

    if mutation_type in ['random3']:    
        i = test[:, 0]  # Shape: (10,)
        j = test[:, 1]  # Shape: (10,)
        k = test[:, 2]  # Shape: (10,)
        l = test[:, 3]
        m = test[:, 4]
        n = test[:, 5]
        o = test[:, 6]
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]
        v5 = x[n]
        v6 = x[o]

    if mutation_type in ['best3']:         
        j = test[:, 0]  # Shape: (10,)
        k = test[:, 1]  # Shape: (10,)
        l = test[:, 2]
        m = test[:, 3]
        n = test[:, 4]
        o = test[:, 5]
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]
        v5 = x[n]
        v6 = x[o]

    if 'best' in mutation_type:
        #base = gen_best
        base = np.full(fill_value=gen_best,shape=(s1,s2,s3))
    else:
        i = test[:, 0]  # Shape: (10,)
        base = x[i]
        base = np.full(fill_value=base,shape=(s1,s2,s3))

    if mutation_type in ['random', 'best']:    
        p = base + F * (v2 - v1) + noise  # Shape: (10, 7, 35)

    if mutation_type in ['random2', 'best2']: 
        p = base + F * (v2 - v1) + F2 * (v4-v3) + noise

    if mutation_type in ['random3', 'best3']: 
        p = base + F * (v2 - v1) + F2 * (v4-v3) + F3 * (v6-v5) + noise 

    return p


def mutate_(NP, NP_indices, F, F2, F3, x, MCMC, mutation_type, gen_best):
    # mcmc noise addition
    b, c = x[0].shape

    # Generate distinct indices for mutation
    indices = np.arange(0, NP)
    test = []
    num_indices = determine_num_indices(mutation_type)
    #print(f'{mutation_type} NP {NP} num_indices {num_indices}')
    test = create_distinct_indices(NP_indices, test, NP, indices, num_indices)
    test = np.array(test)  # Shape: (NP, num_selected), e.g., (10, 3)

    # Create a copy of x to store the result
    y = x.copy()  # Shape: (10, 7, 35)

    # Extract vectors using vectorized indexing
    # Generate noise for all elements at once
    
    noise = generate_noise(b, c, MCMC, (len(NP_indices), b, c))  # Shape eg: (10, 7, 35)
    p = mutate_logic(test, F, F2, F3, x, mutation_type, gen_best, noise,
                      NP, NP_indices)   

    # new
    y = p.copy()

    return y



def generate_noise_old(b,c,MCMC):

    run_mcmc = MCMC.run_mcmc
    error_dist = MCMC.error_dist
    error_std = MCMC.error_std

    if run_mcmc and error_dist == 'norm':
        w = np.random.normal(loc=0, scale=error_std, size=(b,c))

    if run_mcmc and error_dist == 'unif':
        w = np.random.uniform(low= -error_std, high= error_std , size=(b,c))

    if not run_mcmc:
        w = np.zeros((b,c))
    return w


def generate_noise(b,c,MCMC,size_):

    run_mcmc = MCMC.run_mcmc
    error_dist = MCMC.error_dist
    error_std = MCMC.error_std

    if run_mcmc and error_dist == 'norm':
        w = np.random.normal(loc=0, scale=error_std, size=size_)

    if run_mcmc and error_dist == 'unif':
        w = np.random.uniform(low= -error_std, high= error_std , size=size_)

    if not run_mcmc:
        w = np.zeros(size_)
    return w


def crossover_broadcast(NP_indices, y, x, CR, key):

    # determine shape

    z = x.copy()
    m,n = x[0].shape

    # new 

    k_matrix = np.random.uniform(low=0, high=1, size=(len(NP_indices),m,n))
    k_matrix = k_matrix.astype(np.float32)

    # replace 

    condition = k_matrix < CR
    z[condition] = y[condition]

    return z



def return_mutation_current(NP, mutation_type):

    if mutation_type in ['random', 'best']:
        k = 3
    if mutation_type in ['random2', 'best2']:
        k = 5
    if mutation_type in ['random3', 'best3']:
        k = 7
        
    permutations = math.perm(NP-1, k)

    return permutations



def construct_svd_basis(xgp_W0, mindex, maindex, NP_indices):

    sgp_W0 = {}

    for j in NP_indices:
        A = xgp_W0[mindex]
        B = xgp_W0[maindex]
        C = A+B

        U0, S0, V_T0 = svd(A)
        U1, S1, V_T1 = svd(B)
        U2, S2, V_T2 = svd(C)

        w0 = len(S2)       
        S3 = np.zeros((C.shape[0], C.shape[1]))
        S_ = np.diag(S0+S1)
        S3[:w0, :w0] = S_    

        # maybe use a different sv basis?

        D = U2 @ S3 @ V_T2
        sgp_W0[j] = D

    return D


def svd_exploration(NP_indices, current, i_accept, doh, refine_mod,
                       y_train, n_, fitness_metric, error_weight,
                       fitness, X_train, g_points):
    
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = g_points
    
    dgp_W0, dgp_W1, dgp_W2 = {}, {}, {}
    d_errors = []
     
    for k in NP_indices:
        dgp_W0[k] = svd_space(xgp_W0[k],doh)
        dgp_W1[k] = svd_space(xgp_W1[k],doh)
        dgp_W2[k] = svd_space(xgp_W2[k],doh)
        
    # fitness
    
    for k in NP_indices:

        d_rmse, sv = fitness(X_train, dgp_W0[k], dgp_W1[k], dgp_W2[k], xgp_W3[k], xgp_b0[k], xgp_b1[k], xgp_b2[k], xgp_b3[k],
                            y_train, n_, fitness_metric, error_weight)
        d_errors.append(d_rmse)

    # find best fitness

    d_min_value = np.amin(d_errors)
    d_index = np.where(d_errors == d_min_value)
    d_index = d_index[0][0]
    
    best_point = dgp_W0[d_index], dgp_W1[d_index], dgp_W2[d_index], xgp_W3[d_index], xgp_b0[d_index], xgp_b1[d_index], xgp_b2[d_index], xgp_b3[d_index]
        
    return d_min_value, best_point


def return_combo_list(functions, r):    
    master = []        
    l = list(itertools.product(functions, repeat=r))    
    return l


def write_gelman(gb_df, data , run):
    dir_path = '/home/wesley/repos/weights'
    output_loc = os.path.join(dir_path + os.sep, f'GB Diagonistic Run {run}' + '.ods')
    #logging.CRITICAL(f'Saving to {output_loc}')
    with pd.ExcelWriter(output_loc) as writer:
        gb_df.to_excel(writer, sheet_name = f'GR Run {run}', index=False)
        data.to_excel(writer, sheet_name = f'Data Run {run}', index=False)
    return True


def write_fitter(run, fitted):
    dir_path = '/home/wesley/repos/weights'
    output_loc = os.path.join(dir_path + os.sep, f'Fitter Run {run}' + '.ods')
    #logging.CRITICAL(f'Saving to {output_loc}')
    with pd.ExcelWriter(output_loc) as writer:
        fitted.to_excel(writer, sheet_name = f'Fiter Run {run}', index=False)

    return True

def write_weights(W0_, W1_, W2_, W3_, b0_, b1_, b2_, b3_,r, name, X_, y_):
    dir_path = '/home/wesley/repos/weights'
    output_loc = os.path.join(dir_path + os.sep, name + f'index {r}' + '.ods')
    #logging.CRITICAL(f'Saving to {output_loc}')
    with pd.ExcelWriter(output_loc) as writer:
        pd.DataFrame(W0_).to_excel(writer, sheet_name = 'W0', index=False)
        pd.DataFrame(W1_).to_excel(writer, sheet_name = 'W1', index=False)
        pd.DataFrame(W2_).to_excel(writer, sheet_name = 'W2', index=False)
        pd.DataFrame(W3_).to_excel(writer, sheet_name = 'W3', index=False)
        pd.DataFrame(b0_).to_excel(writer, sheet_name = 'b0', index=False)
        pd.DataFrame(b1_).to_excel(writer, sheet_name = 'b1', index=False)
        pd.DataFrame(b2_).to_excel(writer, sheet_name = 'b2', index=False)
        pd.DataFrame(b3_).to_excel(writer, sheet_name = 'b3', index=False)
        pd.DataFrame(X_).to_excel(writer, sheet_name = 'X', index=False)
        pd.DataFrame(y_).to_excel(writer, sheet_name = 'y', index=False)
    return True



def return_refine_count(df):
    cols = ['ClusterCount', 'LocalCount', 'n_SVD_Count', 'scalar_SVD_Count', 'exp_SVD_Count']
    df.loc[(df['clustering_score'] > 0) & (df['clustering_score'] <= df['TrainRMSE']), 'ClusterCount'] = 1
    df.loc[(df['local_score'] > 0) & (df['local_score'] <= df['TrainRMSE']), 'LocalCount'] = 1

    df.loc[(df['svd_value'] > 0) & (df['svd_value'] <= df['TrainRMSE']), 'n_SVD_Count'] = 1
    df.loc[(df['s_scalar_value'] > 0) & (df['s_scalar_value'] <= df['TrainRMSE']), 'scalar_SVD_Count'] = 1
    df.loc[(df['s_exp_value'] > 0) & (df['s_exp_value'] <= df['TrainRMSE']), 'exp_SVD_Count'] = 1
    #df.loc[(df['s_log_value'] > 0) & (df['s_log_value'] <= df['TrainRMSE']), 'log_SVD_Count'] = 1

    final = df.groupby(['Run', 'NP'])[cols].sum()
    return final

def return_standard(dfs, optimum_point, x_2023, NN_model, MCMC, test_data,
                    Data, fitness_metric,models, application, daytype, print_master, DE_model,
                    y_test_true):
    logging.info(f'starting')
    top = [1]
    data = dfs[dfs['Exit'] == 'True'].copy()
    gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3 = optimum_point

    y_2023_pred = DE_model.DENN_forecast(x_2023, gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3, NN_model, MCMC)
    #denn_rmse_test = root_mean_squared_error(test_data[Data.target], y_2023_pred)
    denn_rmse_test = root_mean_squared_error(y_test_true, y_2023_pred)    
    weights=None
    #y_true = np.array(test_data[Data.target],dtype=np.float32)
    y_true = np.array(y_test_true,dtype=np.float32)
    denn_2023_score = return_fitness_metric(y_true, y_2023_pred, fitness_metric, weights)
    data['c'] = 1
    data[f'Test_{fitness_metric}'] = denn_2023_score
    data['Test_RMSE'] = denn_rmse_test
    data['TestStd'] = np.std(y_2023_pred,axis=0)[0]
    models.append(data)

    if print_master:
        xcol = 'datetime'
        label = f'DE-BNN Predicted-{fitness_metric}-{application}'
        file_ext = f'houston-{application}-{daytype}-denn-test-{DE_model.run}.png'
        DE_model.plot(test_data[xcol], Data.target, test_data[Data.target], y_2023_pred, label, file_ext)

    return models, data


def perform_clustering(NP,X_train, y_train, comparison_value, maindex, DE_model,
                       NN_model, n_, gen_points, i, gen_fitness_values):
    #logging.info(f'gen {i} clustering')
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    clustering_list = ['kmeans', 'spectral', 'agg']
    clustering_type = random.choice(clustering_list)

    num_of_clusters_list = list(np.arange(2,NP-2))
    if NP == 4:
        num_of_clusters_list = [3]
    num_of_clusters = random.choice(num_of_clusters_list)
    
    cgp_W0 = cluster_array(xgp_W0, clustering_type, num_of_clusters)
    cgp_W1 = cluster_array(xgp_W1, clustering_type, num_of_clusters)
    cgp_W2 = cluster_array(xgp_W2, clustering_type, num_of_clusters)
    cgp_W3 = cluster_array(xgp_W3, clustering_type, num_of_clusters)

    cgp_b0 = cluster_array(xgp_b0, clustering_type, num_of_clusters)
    cgp_b1 = cluster_array(xgp_b1, clustering_type, num_of_clusters)
    cgp_b2 = cluster_array(xgp_b2, clustering_type, num_of_clusters)
    cgp_b3 = cluster_array(xgp_b3, clustering_type, num_of_clusters)

    clustered = cgp_W0, cgp_W1, cgp_W2, cgp_W3, cgp_b0, cgp_b1, cgp_b2, cgp_b3

    s3,s1,s2 = X_train.shape
    p_ = len(cgp_W0)
    X_trainT = np.full(fill_value=X_train[0],shape=(p_,s1,s2))
    y_train_ = y_train[0,:,:]
    rmse_yp = DE_model.feed(X_trainT, cgp_W0, cgp_W1, cgp_W2, cgp_W3, cgp_b0, cgp_b1, cgp_b2, cgp_b3, n_, NN_model)
    c_errors = np.sqrt(np.mean((rmse_yp - y_train_[None, :, :])**2, axis=(1, 2)))

    exh_den = 1
    tech = 'clustering'
    gen_points = fitness_index_replacement(c_errors, comparison_value, clustered, gen_points, NP, exh_den, DE_model.run, i,
                                            gen_fitness_values, tech)
    return gen_points, min(c_errors)

def perform_search(NP, X_train, y_train, comparison_value, maindex, DE_model,
                        NN_model, n_, gen_points,i, NP_indices, current, gen_fitness_values):

    local_ = 20
    samples = local_ * (int(current/1000) + 1)
    
    #logging.INFO(f'gen {i} STARTING uniform local search samples {samples}')

    # convert random_uniform to 3d array based

    local = random_uniform( gen_points, samples, NP_indices)
    rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3 = local

    s3,s1,s2 = X_train.shape
    p_ = len(rgp_W0)
    X_trainT = np.full(fill_value=X_train[0],shape=(p_,s1,s2))
    y_train_ = y_train[0,:,:]
    rmse_yp = DE_model.feed(X_trainT, rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3, n_, NN_model,)
    l_errors = np.sqrt(np.mean((rmse_yp - y_train_[None, :, :])**2, axis=(1, 2)))

    exh_den = 1
    tech = 'local search'
    gen_points = fitness_index_replacement(l_errors, comparison_value, local, gen_points, NP, exh_den, DE_model.run, i,
                                            gen_fitness_values, tech)

    return gen_points, min(l_errors)

def convert_biases(NP, n_, n1,n2,n3,initial_NP_indices,
                   ix_b0,ix_b1,ix_b2,ix_b3):

    b0_ = np.full((NP, n_, n1), 1.0, dtype=np.float32)
    b1_ = np.full((NP, n_, n2), 1.0, dtype=np.float32)
    b2_ = np.full((NP, n_, n3), 1.0, dtype=np.float32)
    b3_ = np.full((NP, n_, 1), 1.0, dtype=np.float32)

    for j in initial_NP_indices:            
        b0_[j] = ix_b0[j].copy()
        b1_[j] = ix_b1[j].copy()
        b2_[j] = ix_b2[j].copy()
        b3_[j] = ix_b3[j].copy()
    return b0_, b1_, b2_, b3_


def fitness_index_replacement(rmse_ex, min_value, proposed_points, gen_points, NP, exh_den, run, i, gen_fitness_values, tech ):

    p_W0, p_W1, p_W2, p_W3, p_b0, p_b1, p_b2, p_b3 = proposed_points
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    valid_indices = np.where(rmse_ex < min_value)[0]

    # Step 2: Sort the valid indices based on their corresponding rmse_ex values
    sorted_valid_indices = valid_indices[np.argsort(rmse_ex[valid_indices])]

    # Step 3: Select the top 5 indices (or all if fewer than 5)
    # Number of top indices desired
    top_indices = sorted_valid_indices[:min(NP, len(sorted_valid_indices))]
    top_indices = top_indices[:int(NP/exh_den)]
    t = len(top_indices)

    # Step 4: Get corresponding RMSE values for reference
    top_values = rmse_ex[top_indices] if len(top_indices) > 0 else []     
    
    if len(top_indices) > 0:
        logging.info(f'run {run} gen {i} pop min_value {min_value} {tech} min {min(rmse_ex)} replace {t}')

        # Step 1: Ensure gen_fitness_values is 1D; flatten if necessary
        if gen_fitness_values.ndim > 1:
            gen_fitness_values = gen_fitness_values.flatten()

        sorted_indices = np.argsort(gen_fitness_values)

        # Step 3: Select the bottom n indices (lowest values)
        bottom_indices = sorted_indices[:t]

        xgp_W0[bottom_indices] = p_W0[top_indices].copy()
        xgp_W1[bottom_indices] = p_W1[top_indices].copy()
        xgp_W2[bottom_indices] = p_W2[top_indices].copy()
        xgp_W3[bottom_indices] = p_W3[top_indices].copy()

        xgp_b0[bottom_indices] = p_b0[top_indices].copy()
        xgp_b1[bottom_indices] = p_b1[top_indices].copy()
        xgp_b2[bottom_indices] = p_b2[top_indices].copy()
        xgp_b3[bottom_indices] = p_b3[top_indices].copy()

    gen_points = xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 
    return gen_points

def perform_svd_filter(NP, X_train, y_train, comparison_value, maindex, DE_model,
                        NN_model, n_, gen_points,i, NP_indices, current, gen_fitness_values):
    
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3 = {},{},{},{},{},{},{},{}

    # create svd filtered candidates

    S = 0
    for k in NP_indices:
        for j in [1,]:
            dgp_W0[S] = svd_space(xgp_W0[k], j)
            dgp_W1[S] = svd_space(xgp_W1[k], j)
            dgp_W2[S] = svd_space(xgp_W2[k], j)
            dgp_W3[S] = xgp_W3[k]
            dgp_b0[S] = xgp_b0[k]
            dgp_b1[S] = xgp_b1[k]
            dgp_b2[S] = xgp_b2[k]
            dgp_b3[S] = xgp_b3[k]
            S = S+1
    
    # convert dict to numpy 3d array

    dgp_W0 = np.array(list(dgp_W0.values()))    
    dgp_W1 = np.array(list(dgp_W1.values()))  
    dgp_W2 = np.array(list(dgp_W2.values()))
    dgp_W3 = np.array(list(dgp_W3.values()))
    
    dgp_b0 = np.array(list(dgp_b0.values()))    
    dgp_b1 = np.array(list(dgp_b1.values()))  
    dgp_b2 = np.array(list(dgp_b2.values()))
    dgp_b3 = np.array(list(dgp_b3.values())) 

    # fitness

    s3,s1,s2 = X_train.shape
    p_ = len(dgp_W0)
    X_trainT = np.full(fill_value=X_train[0],shape=(p_,s1,s2))
    y_train_ = y_train[0,:,:]
    rmse_yp = DE_model.feed(X_trainT, dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3, n_, NN_model,)
    s_errors = np.sqrt(np.mean((rmse_yp - y_train_[None, :, :])**2, axis=(1, 2)))

    svd_points = dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3

    tech = 'svd_filter'
    exh_den = 1
    gen_points = fitness_index_replacement(s_errors, comparison_value, svd_points, gen_points, NP, exh_den, DE_model.run, i,
                                            gen_fitness_values, tech)

    return gen_points, min(s_errors) 


def perform_svd_scalar(NP, X_train, y_train, comparison_value, maindex, DE_model,
                        NN_model, n_, gen_points,i, NP_indices, current, gen_fitness_values):
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3 = {},{},{},{},{},{},{},{}

    S = 0
    for k in NP_indices:
        U_W0, S_W0, V_T_W0 = svd(xgp_W0[k])
        U_W1, S_W1, V_T_W1 = svd(xgp_W1[k])
        U_W2, S_W2, V_T_W2 = svd(xgp_W2[k])

        for j in np.arange(0,2,0.1):
            dgp_W0[S] = reconstruct_SVD(U_W0, S_W0*j, V_T_W0)
            dgp_W1[S] = reconstruct_SVD(U_W1, S_W1*j, V_T_W1)
            dgp_W2[S] = reconstruct_SVD(U_W2, S_W2*j, V_T_W2)
            dgp_W3[S] = xgp_W3[k]
            dgp_b0[S] = xgp_b0[k]
            dgp_b1[S] = xgp_b1[k]
            dgp_b2[S] = xgp_b2[k]
            dgp_b3[S] = xgp_b3[k]
            S = S+1
    
    # convert dict to numpy 3d array

    dgp_W0 = np.array(list(dgp_W0.values()))    
    dgp_W1 = np.array(list(dgp_W1.values()))  
    dgp_W2 = np.array(list(dgp_W2.values()))
    dgp_W3 = np.array(list(dgp_W3.values()))
    
    dgp_b0 = np.array(list(dgp_b0.values()))    
    dgp_b1 = np.array(list(dgp_b1.values()))  
    dgp_b2 = np.array(list(dgp_b2.values()))
    dgp_b3 = np.array(list(dgp_b3.values())) 

    # fitness

    s3,s1,s2 = X_train.shape
    p_ = len(dgp_W0)
    X_trainT = np.full(fill_value=X_train[0],shape=(p_,s1,s2))
    y_train_ = y_train[0,:,:]
    rmse_yp = DE_model.feed(X_trainT, dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3, n_, NN_model,)
    s_errors = np.sqrt(np.mean((rmse_yp - y_train_[None, :, :])**2, axis=(1, 2)))

    svd_points = dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3

    tech = 'svd_scalar'
    exh_den = 1
    gen_points = fitness_index_replacement(s_errors, comparison_value, svd_points, gen_points, NP, exh_den, DE_model.run, i,
                                            gen_fitness_values, tech)

    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    return gen_points, min(s_errors) 

def perform_svd_exp(NP, X_train, y_train, comparison_value, maindex, DE_model,
                        NN_model, n_, gen_points,i, NP_indices, current, gen_fitness_values):
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3 = {},{},{},{},{},{},{},{}

    S = 0
    for k in NP_indices:
        U_W0, S_W0, V_T_W0 = svd(xgp_W0[k])
        U_W1, S_W1, V_T_W1 = svd(xgp_W1[k])
        U_W2, S_W2, V_T_W2 = svd(xgp_W2[k])

        #for j in np.arange(1.05,1.1,0.01):
        for j in np.arange(1.01,1.2,0.01):
            dgp_W0[S] = reconstruct_SVD(U_W0, S_W0*j, V_T_W0)
            dgp_W1[S] = reconstruct_SVD(U_W1, S_W1*j, V_T_W1)
            dgp_W2[S] = reconstruct_SVD(U_W2, S_W2*j, V_T_W2)
            dgp_W3[S] = xgp_W3[k]
            dgp_b0[S] = xgp_b0[k]
            dgp_b1[S] = xgp_b1[k]
            dgp_b2[S] = xgp_b2[k]
            dgp_b3[S] = xgp_b3[k]
            S = S+1
    
    # convert dict to numpy 3d array

    dgp_W0 = np.array(list(dgp_W0.values()))    
    dgp_W1 = np.array(list(dgp_W1.values()))  
    dgp_W2 = np.array(list(dgp_W2.values()))
    dgp_W3 = np.array(list(dgp_W3.values()))
    
    dgp_b0 = np.array(list(dgp_b0.values()))    
    dgp_b1 = np.array(list(dgp_b1.values()))  
    dgp_b2 = np.array(list(dgp_b2.values()))
    dgp_b3 = np.array(list(dgp_b3.values())) 

    # fitness

    s3,s1,s2 = X_train.shape
    p_ = len(dgp_W0)
    X_trainT = np.full(fill_value=X_train[0],shape=(p_,s1,s2))
    y_train_ = y_train[0,:,:]
    rmse_yp = DE_model.feed(X_trainT, dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3, n_, NN_model,)
    s_errors = np.sqrt(np.mean((rmse_yp - y_train_[None, :, :])**2, axis=(1, 2)))

    svd_points = dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3

    tech = 'svd_filter'
    exh_den = 1
    gen_points = fitness_index_replacement(s_errors, comparison_value, svd_points, gen_points, NP, exh_den, DE_model.run, i,
                                            gen_fitness_values, tech)

    return gen_points, min(s_errors) 


def perform_svd_log(NP, X_train, y_train, comparison_value, maindex, DE_model,
                        NN_model, n_, gen_points,i, NP_indices, current, gen_fitness_values):
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3 = {},{},{},{},{},{},{},{}

    S = 0
    for k in NP_indices:
        U_W0, S_W0, V_T_W0 = svd(xgp_W0[k])
        U_W1, S_W1, V_T_W1 = svd(xgp_W1[k])
        U_W2, S_W2, V_T_W2 = svd(xgp_W2[k])

        for j in np.arange(4,5.05,0.05):
            dgp_W0[S] = reconstruct_SVD(U_W0, np.log( np.diag(S_W0) + 1)**j, V_T_W0)
            dgp_W1[S] = reconstruct_SVD(U_W1, np.log( np.diag(S_W1) + 1)**j, V_T_W1)
            dgp_W2[S] = reconstruct_SVD(U_W2, np.log( np.diag(S_W2) + 1)**j, V_T_W2)
            dgp_W3[S] = xgp_W3[k]
            dgp_b0[S] = xgp_b0[k]
            dgp_b1[S] = xgp_b1[k]
            dgp_b2[S] = xgp_b2[k]
            dgp_b3[S] = xgp_b3[k]
            S = S+1
    
    # convert dict to numpy 3d array

    dgp_W0 = np.array(list(dgp_W0.values()))    
    dgp_W1 = np.array(list(dgp_W1.values()))  
    dgp_W2 = np.array(list(dgp_W2.values()))
    dgp_W3 = np.array(list(dgp_W3.values()))
    
    dgp_b0 = np.array(list(dgp_b0.values()))    
    dgp_b1 = np.array(list(dgp_b1.values()))  
    dgp_b2 = np.array(list(dgp_b2.values()))
    dgp_b3 = np.array(list(dgp_b3.values())) 

    # fitness

    s3,s1,s2 = X_train.shape
    p_ = len(dgp_W0)
    X_trainT = np.full(fill_value=X_train[0],shape=(p_,s1,s2))
    y_train_ = y_train[0,:,:]
    rmse_yp = DE_model.feed(X_trainT, dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3, n_, NN_model,)
    s_errors = np.sqrt(np.mean((rmse_yp - y_train_[None, :, :])**2, axis=(1, 2)))

    svd_points = dgp_W0, dgp_W1, dgp_W2, dgp_W3, dgp_b0, dgp_b1, dgp_b2, dgp_b3

    tech = 'svd_filter'
    exh_den = 1
    gen_points = fitness_index_replacement(s_errors, comparison_value, svd_points, gen_points, NP, exh_den, DE_model.run, i,
                                            gen_fitness_values, tech)

    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    return gen_points, min(s_errors) 

def post_DE(post_de_args, de_output):
    optimum_point, gen_points, dfs, scaler, X_train_scaled, y_train, y_test = de_output
    application, daytype, num_layers, NN_model, Data, test_data, x_2023, fitness_metric,\
                            models,ycol, DE_model, NP_indices, print_master, NP, MCMC, DE_model = post_de_args
    #name = 'DE'
    #success = write_weights( W0_, W1_, W2_, W3_, b0_, b1_, b2_, b3_,run)

    models, data = return_standard(dfs, optimum_point, x_2023, NN_model, MCMC, test_data,
                                    Data, fitness_metric,models, application, daytype, print_master, DE_model,
                                    y_test)    
    return models, data

def plot_concrete(Data, samples_pred, y_test, application, daytype, DE_model):
    boo = Data.testing.copy()
    boo = pd.DataFrame(boo)
    boo['index'] = boo.index
    boo['pred_mean'] = np.mean(samples_pred, axis=0)
    boo['actual'] = y_test
    boo['lower'] = np.percentile(samples_pred, 5, axis=0)
    boo['upper'] = np.percentile(samples_pred, 95, axis=0)
    boohoo = pd.DataFrame(boo, columns = ['index', 'pred_mean', 'actual', 'lower', 'upper'])

    plot_CI_concrete(boohoo, samples_pred, application, daytype, DE_model.run, 'year')


def plot_load(Data, samples_pred, y_test, application, daytype, DE_model):
    dtcol = 'datetime'
    boo = Data.testing.copy()
    boo['pred_mean'] = np.mean(samples_pred, axis=0)
    boo['actual'] = y_test
    boo['lower'] = np.percentile(samples_pred, 5, axis=0)
    boo['upper'] = np.percentile(samples_pred, 95, axis=0)
    boohoo = pd.DataFrame(boo, columns = ['datetime', 'pred_mean', 'actual', 'lower', 'upper'])
    boohoo.index = pd.DatetimeIndex(boohoo.datetime)

    dataset = boohoo.asfreq('h')
    dataset['datetime'] = dataset.index
    dataset['Month'] = dataset['datetime'].dt.month

    plot_CI(dataset, dtcol, samples_pred, application, daytype, DE_model.run, 'year')

    sum_list = [6]
    summer_mask = dataset['Month'].isin(sum_list)
    jdataset = dataset[summer_mask].copy()

    plot_CI(jdataset, dtcol, samples_pred, application, daytype, DE_model.run, 'june')

def posterior_predictive(post_de_mcmc_args, args2, mcmc_chain,G,top_chains, e):
    
    optimum_point, gen_points, dfs, scaler, X_train_scaled, y_train, y_test = args2
    application, daytype, num_layers, NN_model, Data, test_data, x_2023, fitness_metric,\
    ycol, DE_model, NP_indices, print_master, NP, MCMC, DE_model = post_de_mcmc_args

    gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3 = gen_points

    burn_in = MCMC.burn_in
    pred_post_sample = MCMC.pred_post_sample
    multiple_chain = MCMC.multiple_chain
    model_name = application
    
    W0, W1, W2, W3, b0, b1, b2, b3 = mcmc_chain

    data = dfs[dfs['Exit'] == 'True'].copy()
    W0_, W0_fit, W0_T = return_distribution_mode(gb_W0, W0, 'W0', DE_model.run, multiple_chain, MCMC, model_name, top_chains)
    W1_, W1_fit, W1_T = return_distribution_mode(gb_W1, W1, 'W1', DE_model.run, multiple_chain, MCMC, model_name, top_chains)
    W2_, W2_fit, W2_T = return_distribution_mode(gb_W2, W2, 'W2', DE_model.run, multiple_chain, MCMC, model_name, top_chains)
    W3_, W3_fit, W3_T = return_distribution_mode(gb_W3, W3, 'W3', DE_model.run, multiple_chain, MCMC, model_name, top_chains)
        
    b0_, b0_fit, b0_T = return_distribution_mode(gb_b0, b0, 'b0', DE_model.run, multiple_chain, MCMC, model_name, top_chains)
    b1_, b1_fit, b1_T = return_distribution_mode(gb_b1, b1, 'b1', DE_model.run, multiple_chain, MCMC, model_name, top_chains)
    b2_, b2_fit, b2_T = return_distribution_mode(gb_b2, b2, 'b2', DE_model.run, multiple_chain, MCMC, model_name, top_chains)
    b3_, b3_fit, b3_T = return_distribution_mode(gb_b3, b3, 'b3', DE_model.run, multiple_chain, MCMC, model_name, top_chains)

    plot_predictive = True

    # need to batch for larger chain
    # populating such large 3d arrays runs out of memory

    if pred_post_sample not in ['default']:
        total_sample_length = pred_post_sample
    else:
        total_sample_length = len(W0_T)

    print(f'total chain length {total_sample_length} pred_post_sample {pred_post_sample}')
    
    batch_size = 500
    multiple = int(total_sample_length/batch_size)
    num_target = len(y_test)
    samples_pred = np.zeros((total_sample_length,num_target,1), dtype=np.float32)

    for w in np.arange(0,multiple):
        batch_start = w*batch_size
        batch_end = (w+1)*batch_size
        samples_pred_batch = DE_model.DENN_forecast(x_2023, W0_T[batch_start:batch_end], W1_T[batch_start:batch_end], W2_T[batch_start:batch_end], W3_T[batch_start:batch_end],
                                            b0_T[batch_start:batch_end], b1_T[batch_start:batch_end], b2_T[batch_start:batch_end], b3_T[batch_start:batch_end], 
                                            NN_model, MCMC)
        samples_pred[batch_start:batch_end,:,:] = samples_pred_batch

    samples_pred = clean_chain(samples_pred)
    
    # mean of posterior predictive samples
    
    chickenbutt = np.mean(samples_pred[-total_sample_length:,:,:],axis=0)
    rmse_mean_pred = root_mean_squared_error(y_test, chickenbutt)

    #

    boo = Data.testing.copy()
    boo['pred_mean'] = np.mean(samples_pred, axis=0)
    boo['actual'] = y_test
    boo['lower'] = np.percentile(samples_pred, 5, axis=0)
    boo['upper'] = np.percentile(samples_pred, 95, axis=0)

    if plot_predictive:
        if application in ['load']:
            plot_load(Data, samples_pred, y_test, application, daytype, DE_model)

        if application in ['concrete']:
            plot_concrete(Data, samples_pred, y_test, application, daytype, DE_model)

        # computational approximation for weight/bias posterior
        # can use fitted distribution for predictive posterior
        # define distribution and sample from it
        # construct 3d arrays from posterior samples

    # weight/bias mode forecast
    
    test_mode_pred = DE_model.DENN_forecast(x_2023, W0_, W1_, W2_, W3_, b0_, b1_, b2_, b3_, NN_model, MCMC)
    mode_rmse_test = root_mean_squared_error(y_test, test_mode_pred)
    weights=None
    pred_mode_2023_score = return_fitness_metric(y_test, test_mode_pred, fitness_metric, weights)
    data[f'Test_{fitness_metric}_MCMC_mode'] = pred_mode_2023_score
    data['Test_RMSE_MCMC_mode'] = mode_rmse_test
    data['c'] = total_sample_length
    data['num_chain'] = e
    data['pred_post_sample'] = pred_post_sample
    data['TestStd'] = np.std(test_mode_pred,axis=0)[0]
    data['Test_RMSE_MCMC_mean'] = rmse_mean_pred

    rmse_lower = root_mean_squared_error(boo['actual'] , boo['lower'])
    rmse_upper = root_mean_squared_error(boo['actual'] , boo['upper'])

    data['rmse_variance'] = np.minimum(rmse_lower, rmse_upper)

    if multiple_chain and False:
        print('starting gelman rubin')
        # gelman-rubin convergence diagnostic

        gb_df, diag = gelman_rubin(mcmc_chain, gen_points, DE_model.run)
        write_gelman(gb_df, diag, DE_model.run)

    xcol = 'datetime'
    label = f'DE-NN Predicted-MCMC-{fitness_metric}-{application}'
    file_ext = f'{application}-{daytype}-denn-test-{DE_model.run}.png'
    #DE_model.plot(test_data[xcol], ycol, y_test, test_mode_pred, label, file_ext)
    #name='mcmc'
    #success = write_weights( W0_, W1_, W2_, W3_, b0_, b1_, b2_, b3_,run,name)
    
    return data


def post_DE_MCMC(post_de_mcmc_args, args2, mcmc_chain,G, MCMC, models):

    # bayesian neural network
    # single model (set of weight matrics and bias vectors) samples for forecast
    # mode of each posterior

    # posterior predictive sampling
    # setup to go through each multiple chains cumulatively

    top_chains = MCMC.top_chains.copy()
    num_chains = MCMC.chains
    num_chains = 1

    for e in np.arange(0,num_chains):
        chain = top_chains[:e+1]
        data = posterior_predictive(post_de_mcmc_args, args2, mcmc_chain,G, chain, e)
        models.append(data)

    return models


def return_layers(application,daytype,MCMC):

    if daytype == 'weekday':
        layers = (35,10,5)
    if daytype == 'weekend':
        layers = (4,4,3)

    if MCMC.run_mcmc:
        if daytype == 'weekday':
            #layers = (35,25,10)
            layers = (35,10,5)
        if daytype == 'weekend':
            layers = (4,4,3)

    if application == 'concrete':
        layers = (14,12,12)

    return layers

