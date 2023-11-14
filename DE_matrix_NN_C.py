import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import logging
from scipy.linalg import svd
from numpy import array
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import random_projection
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
import ray
from scipy.sparse import random as srandom
from scipy import stats
from numpy.random import default_rng
from sklearn.metrics import mean_squared_error
from scipy.linalg import svd
from scipy.special import expit
from scipy.ndimage import rotate

np.random.seed(42)

# https://en.wikipedia.org/wiki/test_function_names_for_optimization

# classical differential evolution


class NNClass():
    
    def __init__(self, x, y_true, num_layers, n1, n2, n3, activation):
        
        self.x = x
        self.num_layers = num_layers
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.activation = activation
        self.y_true = y_true

        m = len(self.x.columns) # feature dimension
        n = len(y_true) # output dimension

        self.m = m
        self.n = n


    def set_weight(self, key, matrix):
            
        if key == 'W0':
            self.W0 = matrix
        
        if key == 'W1':
            self.W1 = matrix

        if key == 'W2':
            self.W2 = matrix
        
        if key == 'W3':
            self.W3 = matrix

        if key == 'b0':
            self.b0 = matrix

        if key == 'b1':
            self.b1 = matrix

        if key == 'b2':
            self.b2 = matrix

        if key == 'b3':
            self.b3 = matrix
    
    def set_output(self, vector):
            
        self.y_pred = vector


class DEModelClass():
    
    def __init__(self, NP, g, F, CR, mutation_type, clustering_type, 
                 num_of_clusters, cluster_gen_begin, num_replace, angle,
                 tol, rotate_gen_begin, NPI, F_scaling):
        
        self.NP = NP
        self.g = g
        self.F = F
        self.CR = CR
        self.dir_path = r'/home/wesley/repos/data'

        self.mutation_type = mutation_type
        self.clustering_type = clustering_type
        self.num_of_clusters = num_of_clusters
        self.cluster_gen_begin = cluster_gen_begin
        self.rotate_gen_begin = rotate_gen_begin
        self.num_replace = num_replace
        self.angle = angle
        self.tol = tol
        self.NPI = NPI
        self.F_scaling = F_scaling
        self.feed = feed_forward

def elu(w):
    alpha = 10
    w[w >= 0] = w
    w[w < 0] = (expit(w)-1)*alpha
    return w

def selu(w):
    lambda_ = 1.0507
    alpha = 1.6732
    w[w > 0] = w*lambda_
    w[w <= 0] = (expit(w)-1)*alpha*lambda_
    return w


def logistic(x):
    return(1/(1 + np.exp(-x)))

def relu(w):
    w[w >= 0] = w
    w[w < 0] = 0
    return w

def feed_forward(x, W0, W1, W2, W3,
                 b0, b1, b2, b3, y_true, activation):
    
    if activation == 'relu':

        s1 = x@W0 + b0
        z1 = relu(s1)

        s2 = z1@W1 + b1
        z2 = relu(s2)

        s3 = z2@W2 + b2
        z3 = relu(s3)

        y = relu(z3@W3+b3)

    if activation == 'selu':

        s1 = x@W0 + b0
        z1 = selu(s1)

        s2 = z1@W1 + b1
        z2 = selu(s2)

        s3 = z2@W2 + b2
        z3 = selu(s3)

        y = selu(z3@W3+b3)

    if activation == 'tanh':

        s1 = x@W0 + b0
        z1 = np.tanh(s1)

        s2 = z1@W1 + b1
        z2 = np.tanh(s2)

        s3 = z2@W2 + b2
        z3 = np.tanh(s3)

        y = np.tanh(z3@W3+b3)

    if activation == 'logistic':

        s1 = x@W0 + b0
        z1 = logistic(s1)

        s2 = z1@W1 + b1
        z2 = logistic(s2)

        s3 = z2@W2 + b2
        z3 = logistic(s3)

        y = logistic(z3@W3+b3)

    rmse = mean_squared_error(y_true, y, squared=False)
    return rmse

def DENN_forecast(x, W0, W1, W2, W3,
                 b0, b1, b2, b3):

    s1 = x@W0 + b0
    z1 = relu(s1)

    s2 = z1@W1 + b1
    z2 = relu(s2)

    s3 = z2@W2 + b2
    z3 = relu(s3)

    y = relu(z3@W3+b3)
    return y


def construct_matrix(test_indices, weights_, matrix_df):

    for q in np.arange(0,len(matrix_df)):
        test_indices = test_indices.reshape(len(test_indices),1)
        index_to_weight = np.append(test_indices, weights_.reshape(d,1), axis=1) # reshape for testing
        index_to_weight_df = pd.DataFrame(index_to_weight)
        index_to_weight_df = index_to_weight_df.rename(columns = {0:'node_index', 1:'node_weight'})
        # use the indices to put the weights into a dataframe for merging
        
        data = pd.merge(matrix_df, index_to_weight_df, on = ['node_index'], how='left')
        
        del matrix['node_index']
        matrix = matrix.T
        data = np.array(matrix)
        data = np.nan_to_num(data)

        if q == 0:
            diff_matrix = data.copy()
        if q > 0:
            diff_matrix = np.vstack((diff_matrix, data))
    return data

def  cluster_generation(i, xgp, gen_function_value, clustering_type, 
                        num_of_clusters, num_replace, key,
                       f, W0, W1, W2, W3, b0, b1, b2, b3, y_true,
                       fitness):

    # reshaping for sklearn    
    
    a,b = xgp[0].shape
    c = len(xgp)
    k = 0
    h = np.zeros((c,a,b))

    for j in xgp.keys():
        h[k,:a,:b] = xgp[j]
        k = k + 1

    nsamples, nx, ny = h.shape
    d2_train_dataset = h.reshape((nsamples,nx*ny))

    # agglomerative
    # linkage{‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
    if clustering_type == 'agg':
        agg = AgglomerativeClustering(n_clusters=num_of_clusters,linkage='complete',compute_distances=True)
        c_means = agg.fit_predict(d2_train_dataset)
        clf = NearestCentroid()
        clf.fit(d2_train_dataset, c_means)
        centers = clf.centroids_
        clabels = c_means

        plt.title("Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        plot_dendrogram(agg.fit(d2_train_dataset), truncate_mode="level", p=3)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.savefig(f'images/clustering-{clustering_type}-{key}-{i}.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
        plt.close()
        #plt.show()

        # reverse shaping
        centers = centers.reshape(num_of_clusters,nx,ny)

    # testing - find num_replace lowest fitness values for generation

    idx = np.argpartition(gen_function_value, num_replace)
    idx = idx[:num_replace]
    
    # find worst vector and replace with center
    
    max_value = np.amax(gen_function_value)
    worst_idx = np.where(gen_function_value == max_value )
    worst_idx = worst_idx[0][0] # index integer

    # find minimum center value

    center_rmse_list = []
    
    for e in np.arange(0,num_of_clusters):
        if key == 'W0':
            center_rmse = fitness(f, centers[e,:,:], W1, W2, W3, b0, b1, b2, b3, y_true)
        if key == 'W1':
            center_rmse = fitness(f, W0, centers[e,:,:], W2, W3, b0, b1, b2, b3, y_true)
        if key == 'W2':
            center_rmse = fitness(f, W0, W1, centers[e,:,:], W3, b0, b1, b2, b3, y_true)
        if key == 'W3':
            center_rmse = fitness(f, W0, W1, W2, centers[e,:,:], b0, b1, b2, b3, y_true)
        if key == 'b0':
            center_rmse = fitness(f, W0, W1, W2, W3, centers[e,:,:], b1, b2, b3, y_true)
        if key == 'b1':
            center_rmse = fitness(f, W0, W1, W2, W3, b0, centers[e,:,:], b2, b3, y_true)
        if key == 'b2':
            center_rmse = fitness(f, W0, W1, W2, W3, b0, b1, centers[e,:,:], b3, y_true)
        if key == 'b3':
            center_rmse = fitness(f, W0, W1, W2, W3, b0, b1, b2, centers[e,:,:], y_true)
        center_rmse_list.append(center_rmse)

    center_rmses = np.array(center_rmse_list)
    center_rmse_min = np.amin(center_rmses)
    center_result = np.where(center_rmses == center_rmse_min )
    cidx = center_result[0][0]

    # testing - find k highest fitness values for centers

    # center_function_value = test_function_evaluation(centers.T, d)
    # cidx = np.argpartition(center_function_value, num_replace)
    # cidx = cidx[:num_replace]

    # gen worst index, center points, center best index, center best index rmse

    return worst_idx, centers, cidx, center_rmse_min


# def create_cluster(i, xgp, clustering_type, num_of_clusters, key):

#     # reshaping for sklearn    
    
#     a,b = xgp[0].shape
#     c = len(xgp)
#     k = 0
#     h = np.zeros((c,a,b))

#     for j in xgp.keys():
#         h[k,:a,:b] = xgp[j]
#         k = k + 1

#     nsamples, nx, ny = h.shape
#     d2_train_dataset = h.reshape((nsamples,nx*ny))

#     # agglomerative
#     # linkage{‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’

#     if clustering_type == 'agg':
#         agg = AgglomerativeClustering(n_clusters=num_of_clusters,linkage='ward',
#                                       compute_distances=True)
#         c_means = agg.fit_predict(d2_train_dataset)
#         clf = NearestCentroid()
#         clf.fit(d2_train_dataset, c_means)
#         centers = clf.centroids_
#         #clabels = c_means

#         # plt.title("Hierarchical Clustering Dendrogram")
#         # # plot the top three levels of the dendrogram
#         # plot_dendrogram(agg.fit(d2_train_dataset), truncate_mode="level", p=3)
#         # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
#         # plt.savefig(f'images/clustering-{clustering_type}-{key}-{i}.png',
#         #         format='png',
#         #         dpi=300,
#         #         bbox_inches='tight')
#         # plt.close()
#         #plt.show()

#         # reverse shaping
#         centers = centers.reshape(num_of_clusters,nx,ny)

#     return centers

def create_cluster(i, xgp, clustering_type, num_of_clusters, key):

    # reshaping for sklearn    

    # flatten each matrix
    
    d = len(xgp.keys())
    a,b = xgp[0].shape
    c = a*b
    X = np.zeros((c,d))

    for j in xgp.keys():
        X[:,j] = xgp[j].flatten()

    # clustering methods

    if clustering_type == 'kmeans':

        kmeans = KMeans(n_clusters=num_of_clusters, n_init=10, random_state=42)
        kmeans.fit(X.T)   
        c_kmeans = kmeans.predict(X.T)
        centers = kmeans.cluster_centers_
        centers = centers.T
        clabels = c_kmeans

    if clustering_type == 'spectral':
        sc = SpectralClustering(n_clusters=num_of_clusters, n_init=10, affinity='nearest_neighbors', random_state=42).fit(X.T)

        # determine centers from clustering

        df = pd.DataFrame.from_dict({
                'id': list(sc.labels_) ,
                'data': list(X.T)
            })
        #centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).median().agg(np.array,1) original
        centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).mean().agg(np.array,1)
        centers = centers.reset_index(drop = True)
        centers = np.array([np.broadcast_to(row, shape=(c)) for row in centers])
        centers = centers.T
        clabels = sc.labels_

    if clustering_type == 'agg':
        agg = AgglomerativeClustering(n_clusters=num_of_clusters)
        c_means = agg.fit_predict(X.T)
        clf = NearestCentroid()
        clf.fit(X.T, c_means)
        centers = clf.centroids_
        clabels = c_means
        num_of_clusters = len(centers)
        centers = centers.T

    # put centers into a 3d array

    h = np.zeros((num_of_clusters,a,b))
    k = 0
    for j in np.arange(0,num_of_clusters):
        tst = centers[:,j]
        h[k,:a,:b] = tst.reshape((a,b))
        k = k + 1

        # plt.title("Hierarchical Clustering Dendrogram")
        # # plot the top three levels of the dendrogram
        # plot_dendrogram(agg.fit(d2_train_dataset), truncate_mode="level", p=3)
        # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        # plt.savefig(f'images/clustering-{clustering_type}-{key}-{i}.png',
        #         format='png',
        #         dpi=300,
        #         bbox_inches='tight')
        # plt.close()
        #plt.show()

    return h


def plot_cluster(global_min, centers, clustering_type, i, X, clabels):

    plt.figure(figsize=(6,6))
    #plt.xlim(-550,550)
    #plt.ylim(-550,550)
    plt.scatter(X[:, 0], X[:, 1], c=clabels, s=15, cmap='viridis', label='Population')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5, label='Center')
    plt.scatter(global_min[0], global_min[1], c='red', s=150, alpha=0.5, label='Global Minimum')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.title('k-means cluster')
    #plt.legend(fontsize = 8, loc='center right', borderaxespad=0, bbox_to_anchor=(1.02, 1),)
    plt.legend(fontsize = 8, loc='center right')
    plt.savefig(f'images/generation-{clustering_type}-clustering-{i}.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    a = True
    #plt.close()

def constrain_boundary(w, test_function_name):

    if test_function_name == 'eggholder':
        w[w > 512] = random.randrange(-512, 513)
        w[w < -512] = random.randrange(-512, 513)

    if test_function_name == 'rosenbrock':
        w[w > 10] = random.randrange(-5, 10)
        w[w < -5] = random.randrange(-5, 10)
    
    return w


def plot_gen(gen, i, NP):
    plt.plot(gen[0,:,], gen[1,:], '.', markersize=2, label='GenPoint')
    plt.xlim(-550,550)
    plt.ylim(-550,550)
    plt.xlabel('x')
    plt.ylabel('y')
    kfc = ('Generation Point', str(NP), str(i))
    plt.title(kfc )
    plt.legend(fontsize = 5)
    output_name = ' '.join(kfc)
    #output_loc = os.path.join(dir_path + os.sep + 'gen', output_name + '.png')
    #plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    plt.show()
    plt.close()

def create_extrema_dict(d, test):
    min_ = {}
    max_ = {}

    if test == 'poisson':
        for j in np.arange(0,d):
            min_[j] = -5
        for j in np.arange(0,d):
            max_[j] = 10

    return min_, max_

def generate_initial_population(a, b, NP_indices, key):

    candidates = {}

    # bias matrix initialization

    # Kaiming He weight initialization for relu
    # loc = mean, scale = standard deviation

    std = np.sqrt(2.0 / a)

    if key in ['WO', 'W1', 'W2', 'W3']:
        for j in NP_indices:
            x = np.random.normal(loc=0, scale=std, size=(a,b))
            candidates[j] = x
    else:
        for j in NP_indices:
            x = np.random.normal(loc=0, scale=std, size=(a,b))
            candidates[j] = x
    return candidates

def mutate(NP, NP_indices, F, x):
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 2, replace=False)
        a = list(a)
        a.insert(0, j )
        test.append(a)
        indices = list(np.arange(0,NP))

    y = x.copy()

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        base = x[i]
        v1 = x[j]
        v2 = x[k]
        p = base + F*(v2-v1)
        y[e] = p

    return y

def mutate_best(NP, NP_indices, F, gen_best, x):
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 2, replace=False)
        a = list(a)
        a.insert(0, j )
        test.append(a)
        indices = list(np.arange(0,NP))

    y = x.copy()

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        base = gen_best
        v1 = x[j]
        v2 = x[k]
        p = base + F*(v2-v1)
        y[e] = p

    return y

def crossover(a, NP, NP_indices, y, x, CR):

    z = x.copy()

    for e in NP_indices:
        k = np.random.choice(np.arange(0,a),)
        for i in np.arange(0,a):
            if (random.uniform(0, 1) <= CR or i == k):
                z[e][i] = y[e][i]
            else:
                z[e][i] = x[e][i]
    return z

def selection(NP_indices, fitness, NN_model,
              x_W0, z_W0, x_W1, z_W1, x_W2, z_W2, x_W3, z_W3,
              x_b0, z_b0, x_b1, z_b1, x_b2, z_b2, x_b3, z_b3):

    # latest generation's weight and bias matrices
    activation = NN_model.activation
    f = NN_model.x
    y_true = NN_model.y_true

    # determine survival of target or trial vector
    # into the next generation
    
    for j in NP_indices:
        if fitness(f, z_W0[j], z_W1[j], z_W2[j], z_W3[j], z_b0[j], z_b1[j], z_b2[j], z_b3[j], y_true, activation) <= fitness(f, x_W0[j], x_W1[j], x_W2[j], x_W3[j], x_b0[j], x_b1[j], x_b2[j], x_b3[j], y_true, activation): 
            x_W0[j] = z_W0[j]
            x_W1[j] = z_W1[j]
            x_W2[j] = z_W2[j]
            x_W3[j] = z_W3[j]
            x_b0[j] = z_b0[j]
            x_b1[j] = z_b1[j]
            x_b2[j] = z_b2[j]
            x_b3[j] = z_b3[j]
        else:
            x_W0[j] = x_W0[j]
            x_W1[j] = x_W1[j]
            x_W2[j] = x_W2[j]
            x_W3[j] = x_W3[j]
            x_b0[j] = x_b0[j]
            x_b1[j] = x_b1[j]
            x_b2[j] = x_b2[j]
            x_b3[j] = x_b3[j]
    return x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3   

#@ray.remote
def differential_evolution(DE_model, rmse_values, a, b, NN_model):

    #print(f'starting {DE}')
    
    NP = DE_model.NP
    g = DE_model.g
    F = DE_model.F
    CR = DE_model.CR
    fitness = DE_model.feed
    mutation_type = DE_model.mutation_type

    # clustering parameters

    clustering_type = DE_model.clustering_type
    num_of_clusters = DE_model.num_of_clusters
    cluster_gen_begin = DE_model.cluster_gen_begin
    rotate_gen_begin = DE_model.rotate_gen_begin
    num_replace = DE_model.num_replace
    angle = DE_model.angle
    tol = DE_model.tol
    NPI = DE_model.NPI
    F_scaling = DE_model.F_scaling

    ####

    # latest generation's weight and bias matrices

    activation = NN_model.activation
    f = NN_model.x
    y_true = NN_model.y_true

    ####

    m = NN_model.m
    n = NN_model.n
    n1 = NN_model.n1
    n2 = NN_model.n2
    n3 = NN_model.n3

    #

    NP_indices = list(np.arange(0,NP))
    initial_NP_indices = list(np.arange(0,NPI))
    df_list=[]

    for i in np.arange(0,g):

        # generate initial population for each weight matrix      

        if i == 0:
            x_W0 = generate_initial_population(m, n1, initial_NP_indices, 'W0')
            x_W1 = generate_initial_population(n1, n2, initial_NP_indices, 'W1')
            x_W2 = generate_initial_population(n2, n3, initial_NP_indices, 'W2')
            x_W3 = generate_initial_population(n3, 1, initial_NP_indices, 'W3')

            x_b0 = generate_initial_population(n, n1, initial_NP_indices, 'b0')
            x_b1 = generate_initial_population(n, n2, initial_NP_indices, 'b1')
            x_b2 = generate_initial_population(n, n3, initial_NP_indices, 'b2')
            x_b3 = generate_initial_population(n, 1, initial_NP_indices, 'b3')

            gen_best_x_W0 = x_W0[0]
            gen_best_x_W1 = x_W1[0]
            gen_best_x_W2 = x_W2[0]
            gen_best_x_W3 = x_W3[0]

            gen_best_x_b0 = x_b0[0]
            gen_best_x_b1 = x_b1[0]
            gen_best_x_b2 = x_b2[0]
            gen_best_x_b3 = x_b3[0]

            residual_tracking = []

            # initial population fitness

            initial_fitness = []

            for j in initial_NP_indices:
                init_rmse = fitness(f, x_W0[j], x_W1[j], x_W2[j], x_W3[j], x_b0[j], x_b1[j], x_b2[j], x_b3[j], y_true, activation)
                initial_fitness.append(init_rmse)

            # find best initial generation candidates

            #iidx = np.argpartition(initial_fitness, -NP)[-num_replace:]
            iidx = np.argpartition(initial_fitness, NP-1)[:NP]

            for j in NP_indices:
                x_W0[j] = x_W0[iidx[j]]
                x_W1[j] = x_W1[iidx[j]]
                x_W2[j] = x_W2[iidx[j]]
                x_W3[j] = x_W3[iidx[j]]

                x_b0[j] = x_b0[iidx[j]]
                x_b1[j] = x_b1[iidx[j]]
                x_b2[j] = x_b2[iidx[j]]
                x_b3[j] = x_b3[iidx[j]]
            
            initial_fitness.sort()
            w = np.mean(initial_fitness[:NP])
            print(f'best fitness is {initial_fitness[0]}, avg fitness {w}, NPI {NPI}')

        if i > 0:
            gen_best_x_W0 = gb_W0
            gen_best_x_W1 = gb_W1
            gen_best_x_W2 = gb_W2
            gen_best_x_W3 = gb_W3

            gen_best_x_b0 = gb_b0
            gen_best_x_b1 = gb_b1
            gen_best_x_b2 = gb_b2
            gen_best_x_b3 = gb_b3

        # mutation
        
        # DE/rand/1

        if mutation_type == 'random':

            y_W0 = mutate(NP, NP_indices, F, x_W0)
            y_W1 = mutate(NP, NP_indices, F, x_W1)
            y_W2 = mutate(NP, NP_indices, F, x_W2)
            y_W3 = mutate(NP, NP_indices, F, x_W3)

            y_b0 = mutate(NP, NP_indices, F, x_b0)
            y_b1 = mutate(NP, NP_indices, F, x_b1)
            y_b2 = mutate(NP, NP_indices, F, x_b2)
            y_b3 = mutate(NP, NP_indices, F, x_b3)
        
        # DE/best/1

        if mutation_type in ['best', 'gen_best']:

            y_W0 = mutate_best(NP, NP_indices, F, gen_best_x_W0, x_W0)
            y_W1 = mutate_best(NP, NP_indices, F, gen_best_x_W1, x_W1)
            y_W2 = mutate_best(NP, NP_indices, F, gen_best_x_W2, x_W2)
            y_W3 = mutate_best(NP, NP_indices, F, gen_best_x_W3, x_W3)

            y_b0 = mutate_best(NP, NP_indices, F, gen_best_x_b0, x_b0)
            y_b1 = mutate_best(NP, NP_indices, F, gen_best_x_b1, x_b1)
            y_b2 = mutate_best(NP, NP_indices, F, gen_best_x_b2, x_b2)
            y_b3 = mutate_best(NP, NP_indices, F, gen_best_x_b3, x_b3)

        # crossover

        z_W0 = crossover(m, NP, NP_indices, y_W0, x_W0, CR)
        z_W1 = crossover(n1, NP, NP_indices, y_W1, x_W1, CR)
        z_W2 = crossover(n2, NP, NP_indices, y_W2, x_W2, CR)
        z_W3 = crossover(n3, NP, NP_indices, y_W3, x_W3, CR)

        z_b0 = crossover(m, NP, NP_indices, y_b0, x_b0, CR)
        z_b1 = crossover(n1, NP, NP_indices, y_b1, x_b1, CR)
        z_b2 = crossover(n2, NP, NP_indices, y_b2, x_b2, CR)
        z_b3 = crossover(n3, NP, NP_indices, y_b3, x_b3, CR)

        # selection 
        
        xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = selection(NP_indices, fitness, NN_model, 
                                                   x_W0, z_W0, x_W1, z_W1, x_W2, z_W2, x_W3, z_W3,
                                                   x_b0, z_b0, x_b1, z_b1, x_b2, z_b2, x_b3, z_b3)
        
        # fitness evaluation

        errors = []

        for j in NP_indices:
            gen_rmse = fitness(f, xgp_W0[j], xgp_W1[j], xgp_W2[j], xgp_W3[j], xgp_b0[j], xgp_b1[j], xgp_b2[j], xgp_b3[j], y_true, activation)
            errors.append(gen_rmse)
            
        # determine best generation point

        gen_fitness_values = np.array(errors)
        min_value = np.amin(gen_fitness_values)
        mindex = np.where(gen_fitness_values == np.amin(min_value))
        mindex = mindex[0][0] # index integer
        rmse_values.append(min_value)
        logging.info(f'gen {i} index {mindex} min rmse {min_value}')
        gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3 = xgp_W0[mindex], xgp_W1[mindex], xgp_W2[mindex], xgp_W3[mindex], xgp_b0[mindex], xgp_b1[mindex], xgp_b2[mindex], xgp_b3[mindex]

        # determine worst generation point

        max_value = np.amin(gen_fitness_values)
        maindex = np.where(gen_fitness_values == np.amin(max_value))
        maindex = maindex[0][0]

        # track residual
        
        residual = rmse_values[i]-rmse_values[i-1]
        residual_tracking.append(residual)
        avg_residual = sum(residual_tracking[-10:])/len(residual_tracking[-10:])

        # running best

        if i == 0:
        #if i == 0 and mutation_type in ['gen_best']:
            logging.info(f'gen {i} setting running best')
            rb_W0, rb_W1, rb_W2, rb_W3, rb_b0, rb_b1, rb_b2, rb_b3 = xgp_W0[mindex], xgp_W1[mindex], xgp_W2[mindex], xgp_W3[mindex], xgp_b0[mindex], xgp_b1[mindex], xgp_b2[mindex], xgp_b3[mindex]
            running_min_value = min_value

        if i > 0 and min_value < running_min_value:
        #if i > 0 and min_value < running_min_value and mutation_type == 'gen_best':
            logging.info(f'gen {i} updating running best')
            running_min_value = min_value
            rb_W0, rb_W1, rb_W2, rb_W3, rb_b0, rb_b1, rb_b2, rb_b3 = xgp_W0[mindex], xgp_W1[mindex], xgp_W2[mindex], xgp_W3[mindex], xgp_b0[mindex], xgp_b1[mindex], xgp_b2[mindex], xgp_b3[mindex]

        # rotation

        F = DE_model.F

        if residual >= tol and i >= rotate_gen_begin and angle is not None: # and i < 300:
            logging.info(f'starting rotation gen {i}')
            num_replace = DE_model.num_replace
            
            if avg_residual > 0:
                num_replace = num_replace*2
                F = DE_model.F*F_scaling
                logging.info(f'rotation gen {i} num replace doubled {num_replace} F {F}')
            else:
                num_replace = DE_model.num_replace

            xy = np.argpartition(gen_fitness_values, -num_replace)[-num_replace:]
            rot_angle = angle
            #scipy.linalg.subspace_angles(a,b)
            #c = np.rot90(a, k=1, axes=(0, 1))

            for q in xy:
                #xgp_W0[q] = rotate(xgp_W0[q], angle=rot_angle, reshape=False) # handle vectors differently? 
                xgp_W0[q] = rotate(xgp_W0[q], angle=rot_angle, reshape=True).T
                xgp_W1[q] = rotate(xgp_W1[q], angle=rot_angle, reshape=False) # okay to not reshape with square matrices!
                xgp_W2[q] = rotate(xgp_W2[q], angle=rot_angle, reshape=False)
                xgp_W3[q] = rotate(xgp_W3[q], angle=rot_angle, reshape=True).T # handle vectors differently? 
                xgp_b0[q] = rotate(xgp_b0[q], angle=rot_angle, reshape=False)
                xgp_b1[q] = rotate(xgp_b1[q], angle=rot_angle, reshape=False)
                xgp_b2[q] = rotate(xgp_b2[q], angle=rot_angle, reshape=False)
                xgp_b3[q] = rotate(xgp_b3[q], angle=rot_angle, reshape=False)
        
        # clustering

        if residual >= tol and i >= cluster_gen_begin and clustering_type is not None:
            logging.info(f'starting clustering gen {i}')
            W0_centers = create_cluster(i, xgp_W0, clustering_type, num_of_clusters, 'W0')
            W1_centers = create_cluster(i, xgp_W1, clustering_type, num_of_clusters, 'W1')
            W2_centers = create_cluster(i, xgp_W2, clustering_type, num_of_clusters, 'W2')
            W3_centers = create_cluster(i, xgp_W3, clustering_type, num_of_clusters, 'W3')

            b0_centers = create_cluster(i, xgp_b0, clustering_type, num_of_clusters, 'b0')
            b1_centers = create_cluster(i, xgp_b1, clustering_type, num_of_clusters, 'b1')
            b2_centers = create_cluster(i, xgp_b2, clustering_type, num_of_clusters, 'b2')
            b3_centers = create_cluster(i, xgp_b3, clustering_type, num_of_clusters, 'b3')

            # find clusters with best fitness values

            cluster_errors = []

            for j in np.arange(0,num_of_clusters):
                cluster_rmse = fitness(f, W0_centers[j,:,:], W1_centers[j,:,:], W2_centers[j,:,:], W3_centers[j,:,:], 
                                       b0_centers[j,:,:], b1_centers[j,:,:], b2_centers[j,:,:], b3_centers[j,:,:], y_true, activation)
                cluster_errors.append(cluster_rmse)

            center_minx_value = np.amin(cluster_errors)
            center_index = np.where(cluster_errors == np.amin(center_minx_value))
            center_index = center_index[0][0]
            logging.info(f'center min rmse {center_minx_value}')

            # find worst generation candidates

            idx = np.argpartition(gen_fitness_values, -num_replace)[-num_replace:]

            # find best cluster candidates

            cidx = np.argpartition(cluster_errors, num_replace)
            cidx = cidx[:num_replace]

            # replace indices

            for foo, bar in zip(idx, cidx):
                xgp_W0[foo] = W0_centers[bar,:,:]
                xgp_W1[foo] = W1_centers[bar,:,:]
                xgp_W2[foo] = W2_centers[bar,:,:]
                xgp_W3[foo] = W3_centers[bar,:,:]

                xgp_b0[foo] = b0_centers[bar,:,:]
                xgp_b1[foo] = b1_centers[bar,:,:]
                xgp_b2[foo] = b2_centers[bar,:,:]
                xgp_b3[foo] = b3_centers[bar,:,:]

        df = pd.DataFrame({'Generation':[i], 'F':[F], 'CR':[CR], 'G':[g], 'NP':[NP], 'mutation_type':[mutation_type], 
                           'clustering':[clustering_type], 'num_of_clusters':[num_of_clusters], 'cluster_gen_begin':[cluster_gen_begin], 
                    'rotate_gen_begin':[rotate_gen_begin], 'num_replace':[num_replace], 
                    'angle':[angle], 'tol':[tol], 'RMSE':[min_value], 'rRMSE':[running_min_value], 
                    'residual':[residual], 'run_avg_residual':[avg_residual], 'NPI':[NPI], 'F_scaling':[F_scaling]
                    })
        df_list.append(df)
    
    optimum_point = gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3
    dfs = pd.concat(df_list, sort = False)
    return optimum_point, dfs, rmse_values
    #return dfs


from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)