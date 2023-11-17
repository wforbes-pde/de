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

random.seed(42)

# https://en.wikipedia.org/wiki/test_function_names_for_optimization

# classical differential evolution


class DEModelClass():
    
    def __init__(self, d, NP, g, F, CR, test_function_name, tol,
                 mutation_type, clustering_type, num_of_clusters, cluster_gen_begin, num_replace):
        
        self.d = d
        self.NP = NP
        self.g = g
        self.F = F
        self.CR = CR
        self.test_function_name = test_function_name
        self.tol = tol
        self.global_min = return_vtr(self.test_function_name, self.d)
        self.min_, self.max_ = create_extrema_dict(self.d, self.test_function_name)
        self.dir_path = r'/home/wesley/repos/data'

        self.mutation_type = mutation_type
        self.clustering_type = clustering_type
        self.num_of_clusters = num_of_clusters
        self.cluster_gen_begin = cluster_gen_begin
        self.num_replace = num_replace

        if self.test_function_name == 'eggholder':
            self.test_function_selection = eggholder
            self.test_function_evaluation = eggholder_eval

        if self.test_function_name == 'rosenbrock':
            self.test_function_selection = rosenbrock
            self.test_function_evaluation = rosenbrock_eval

def cluster_generation(i, d, xgp, gen_function_value, global_min, test_function_evaluation,
                       clustering_type, num_of_clusters, num_replace):

    X = xgp.T

    # predetermined number of clusters for kmeans, spectral

    if clustering_type == 'kmeans':

        kmeans = KMeans(n_clusters=num_of_clusters, n_init=10, random_state=42)
        kmeans.fit(X)        
        c_kmeans = kmeans.predict(X)
        centers = kmeans.cluster_centers_
        clabels = c_kmeans

    # spectral clustering

    if clustering_type == 'spectral':
        sc = SpectralClustering(n_clusters=num_of_clusters, n_init=5, affinity='nearest_neighbors', random_state=42).fit(X)

        # determine centers from clustering

        df = pd.DataFrame.from_dict({
                'id': list(sc.labels_) ,
                'data': list(X)
            })
        #centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).median().agg(np.array,1) original
        centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).mean().agg(np.array,1)
        centers = centers.reset_index(drop = True)
        centers = np.array([np.broadcast_to(row, shape=(d)) for row in centers])
        clabels = sc.labels_

    # affinity

    if clustering_type == 'affinity':
        ap = AffinityPropagation(random_state=42)
        c_means = ap.fit(X)
        centers = ap.cluster_centers_
        clabels = c_means    
    
    # agglomerative

    if clustering_type == 'agg':
        agg = AgglomerativeClustering(n_clusters=num_of_clusters)
        c_means = agg.fit_predict(X)
        clf = NearestCentroid()
        clf.fit(X, c_means)
        centers = clf.centroids_
        clabels = c_means

    # no predetermined number of clusters for mean shift, dbscan
    
    if clustering_type == 'mean_shift':
        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
        mean_shift = MeanShift(bandwidth=bandwidth, cluster_all = False)
        mean_shift.fit(X)
        c_means = mean_shift.predict(X)
        centers = mean_shift.cluster_centers_
        clabels = c_means

    # dimensionality reduction, then use kmeans
    
    if clustering_type == 'rand_proj':
        transformer = random_projection.SparseRandomProjection(random_state=42, eps=0.75)
        X_new = transformer.fit_transform(xgp)
        print(X_new.shape)

    if clustering_type == 'nmf':
        nmf = NMF(n_components=2, init='random', random_state=42)
        X_new = nmf.fit_transform(xgp)
        print(X_new.shape)

    if False and i == 50:
        plot_cluster(global_min, centers, clustering_type, i, X, clabels)
    
    # testing - find num_replace lowest fitness values for generation

    idx = np.argpartition(gen_function_value, num_replace)
    idx = idx[:num_replace]
    
    # find worst vector and replace with center
    
    max_value = np.amax(gen_function_value)
    resultm = np.where(gen_function_value == np.amax(gen_function_value))
    resultm = resultm[0][0] # index integer

    # find minimum center value

    center_function_value = test_function_evaluation(centers.T, d)
    center_result = np.where(center_function_value == np.amin(center_function_value))
    center_result = center_result[0][0] # index integer

    # testing - find k highest fitness values for centers

    center_function_value = test_function_evaluation(centers.T, d)
    cidx = np.argpartition(center_function_value, num_replace)
    cidx = cidx[:num_replace]

    # gen worst index, center points, center best index

    return idx, centers, cidx


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

    # replace boundary violations with random domain values

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

def plot_re_sv_gen(x, values, svd_list):
    
    y1 = values
    y2 = svd_list[0:][0]
    y2 = y2[1:]

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots(figsize=(26,16))
    plt.title('function and singular value versus generation')

    # make a plot
    ax.plot(x,
            y1,
            color="dodgerblue")
    # set x-axis label
    ax.set_xlabel("generation", fontsize = 10)
    # set y-axis label
    ax.set_ylabel("log relative error",
                color="dodgerblue",
                fontsize=10)

    ax2=ax.twinx()    
    # make a plot with different y-axis using second axis object
    ax2.plot(x, 
                y2,color="brown")
    ax2.set_ylabel("singular value",color="brown",fontsize=8)

    #plt.show()

    # save the plot as a file
    fig.savefig('images/error-singular-gen.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.close()

def plot_fv_spectral_gen(x, values, spectral_values):

    y1 = values
    y4 = spectral_values

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots(figsize=(26,16))
    plt.title('function and spectral value versus generation')

    # make a plot
    ax.plot(x,
            y1,
            color="dodgerblue")
    # set x-axis label
    ax.set_xlabel("generation", fontsize = 10)
    # set y-axis label
    ax.set_ylabel("function value",
                color="dodgerblue",
                fontsize=10)
    
    ax2=ax.twinx()    
    # make a plot with different y-axis using second axis object
    ax2.plot(x, 
                y4,color="brown")
    ax2.set_ylabel("spectral value",color="brown",fontsize=8)
    
    #plt.show()

    # save the plot as a file
    fig.savefig('images/value-spectral-gen.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.close()


def plot_svd(m_, j, dir_path, name, svd_list):
    # store eigenvalues
    
    #new = np.linalg.eigvals(a)
    #eig_list[j] = new
    
    # store singular valvues
    
    # Singular-value decomposition    
    
    # SVD

    U, S, V_T = svd(m_)

    # store singular values
    #svd_list[j] = S
    svd_list = np.hstack((svd_list, S.reshape(len(S),1)))
    
    # left singular vectors
    #print("U=")
    #print(U)
    
    # singular values
    #print("S=")
    #print(S)
    
    #right singular vectors
    
    #print("V_T=")
    #print(V_T)

    # condition number from sv)
        
    sigma_1 = S[0]
    sigma_n = S[len(S)-1]
    
    cond_num = sigma_1/sigma_n
    
    v = np.arange(0, len(m_))
    plt.plot(v, S, '.', label='Singular Value')
    #plt.ylim(0,22)
    #plt.xlim(0,9)
    plt.xlabel('index')
    plt.ylabel('singular value')
    plt.title('Population Matrix Singular Value ' + name)
    plt.legend()
    #plt.show()

    subfolder = os.path.join(dir_path + os.sep + 'svd')
        
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)  
    
    kfc = ('Population Matrix Singular Value', name)
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep + 'svd', output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    #plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    plt.close()

    return svd_list, cond_num

def plot_evd(matrix, i, dir_path, name, evd_list):
    
    new = np.linalg.eigvals(matrix)
    a = new.real
    b = new.imag
    spectral_radius = np.sqrt(a**2 + b**2)
    plt.plot(new.real, new.imag, '.', label='Eigenvalue')
    plt.ylim(-10,10)
    plt.xlim(-10,10)
    plt.xlabel('real axis')
    plt.ylabel('imaginary axis')
    kfc = ('Eigenvalue Spectrum', name)
    output_name = ' '.join(kfc)
    plt.title(output_name)
    plt.legend()
    #plt.show()
    
    subfolder = os.path.join(dir_path + os.sep + 'evd')
    
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)    
    
    output_loc = os.path.join(dir_path + os.sep + 'evd', output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    #plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    plt.close()
    return spectral_radius.max()


def rosenbrock(p, d): # used in selection operator
    # population component wise
    p = p.reshape(len(p),1)
    x_i = p[0:d-1,:] # range is 1 to d-1
    x_pi = p[1:d,:] # range is 2 to d

    a = (x_i-1)**2
    b = 100*(x_pi - x_i**2)**2
    c = a + b
    f = np.sum(c, axis=0)
    return f

def rosenbrock_eval(p, d):
    # candidate vector wise    
    x_i = p[0:d-1,:] # range is 1 to d-1
    x_pi = p[1:d,:] # range is 2 to d

    a = (x_i-1)**2
    b = 100*(x_pi - x_i**2)**2
    c = a + b
    f = np.sum(c, axis=0)
    return f

def eggholder(p, d): # used in selection operator
    # population component wise
    p = p.reshape(len(p),1)
    a = p[1,:] + 47
    b = p[1,:] + p[0,:]/2 + 47
    c = p[0,:] - (p[1,:] +47)
    f = -a * np.sin(np.sqrt(np.abs(b) ) ) - p[0,:] * np.sin(np.sqrt(np.abs(c) ) )
    return f

def eggholder_eval(p, d):
    # candidate vector wise
    
    a = p[1,:] + 47
    b = p[1,:] + p[0,:]/2 + 47
    c = p[0,:] - (p[1,:] +47)
    f = -a * np.sin(np.sqrt(np.abs(b) ) ) - p[0,:] * np.sin(np.sqrt(np.abs(c) ) )
    return f

def styb_tang(p, d):
    # population component wise
    
    a = p**4
    b = 16*p**2
    c = 5*p

    f = (a.sum() - b.sum() + c.sum())/2
    return f

def styb_tang_eval(p, d):
    # candidate vector wise
    
    a = p**4
    b = 16*p**2
    c = 5*p
    x = a.sum(axis=0)
    y = b.sum(axis=0)
    z = c.sum(axis=0)

    f = ( x - y + z)/2
    return f

def beale(p):
    # population component wise

    x = p[0]
    y = p[1]
    a = (1.5 - x + x*y)**2
    b = (2.25 - x + x*y**2)**2
    c = (2.625 - x + x*y**3)**2

    f = a + b + c
    return f

def beale_eval(x):
    # candidate vector wise
    
    x_ = x[:,0]
    y_ = x[:,1]
    a = (1.5 - x_ + x_*y_)**2
    b = (2.25 - x_ + x_*y_**2)**2
    c = (2.625 - x_ + x_*y_**3)**2

    f = a + b + c
    return f

def create_extrema_dict(d, test):
    min_ = {}
    max_ = {}
    if test == 'styb_tang':
        for j in np.arange(0,d):
            min_[j] = -5
        for j in np.arange(0,d):
            max_[j] = 5

    if test == 'eggholder':
        for j in np.arange(0,d):
            min_[j] = -512
        for j in np.arange(0,d):
            max_[j] = 512

    if test == 'rosenbrock':
        for j in np.arange(0,d):
            min_[j] = -5
        for j in np.arange(0,d):
            max_[j] = 10

    return min_, max_

def return_vtr(test_function_name, d):
    vtr_dict = {}
    vtr_dict['beale'] = 1e-6

    if test_function_name == 'styb_tang' and False:
        lower = -39.16617*d 
        upper = -39.16616*d
        vtr_dict['styb_tang'] = lower, upper
    
    if test_function_name == 'styb_tang':
        g = -2.903534
        x = np.full((d, 1), g)
        vtr_dict['styb_tang'] = x

    if test_function_name == 'eggholder':
        x = np.array( [512, 404.2319])
        vtr_dict['eggholder'] = x

    if test_function_name == 'rosenbrock':
        g = 1
        x = np.full((d, 1), g)
        vtr_dict['rosenbrock'] = x

    return vtr_dict[test_function_name]

def generate_initial_population(d, NP, test_function_name, min_, max_):

    # create initial population based on dimension d and number of 
    # candidates NP and prescribed minimum and maximum values in
    # each component

    x =  np.full((d, NP), 1.0)

    # initital population

    if test_function_name in ['styb_tang', 'eggholder', 'rosenbrock']:

        for i in np.arange(0,NP):
            for j in np.arange(0,d):
                x[j,i] = min_[j] + random.uniform(0, 1) * (max_[j] - min_[j] )

    return x

def mutate(d, NP, NP_indices, F, x):

    # mutate mutation with distinct indices
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 2, replace=False)
        a = list(a)
        a.insert(0, j )
        test.append(a)
        indices = list(np.arange(0,NP))

    y = np.full((d, NP), 1.0)

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        base = x[:,i]
        v1 = x[:,j]
        v2 = x[:,k]
        p = base + F*(v2-v1)
        y[:,e] = p

    return y

def mutate_best(d, NP, NP_indices, F, gen_best, x):

    # best mutation with distinct indices
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 2, replace=False)
        a = list(a)
        a.insert(0, j )
        test.append(a)
        indices = list(np.arange(0,NP))

    y = np.full((d, NP), 5.0)

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        base = gen_best.reshape(d,)
        v1 = x[:,j]
        v2 = x[:,k]
        p = base + F*(v2-v1)
        y[:,e] = p

    return y

def crossover(d, NP, NP_indices, y, x, CR):

    # crossover with at least one dimension swapped based on random int k

    z = np.full((d, NP), 1.0)

    for e in NP_indices:
        k = np.random.choice(np.arange(0,d),)
        for i in np.arange(0,d):
            if (random.uniform(0, 1) <= CR or i == k):
                z[i][e] = y[i][e]
            else:
                z[i][e] = x[i][e]
    return z

def selection(d, NP_indices, x, z, test_function_def):

    # determine survival of target or trial vector
    # into the next generation

    for e in NP_indices:
        if test_function_def(z[:,e],d) <= test_function_def(x[:,e],d):
            x[:,e] = z[:,e]
        else:
            x[:,e] = x[:,e]
    return x, z


def differential_evolution(DE_model, values, cond_num_values, spectral_values, relative_error_values, rank_values, 
                           svd_list, evd_list, eggholder):

    d = DE_model.d
    NP = DE_model.NP
    g = DE_model.g
    F = DE_model.F
    CR = DE_model.CR
    test_function_name = DE_model.test_function_name
    tol = DE_model.tol
    min_ = DE_model.min_
    max_ = DE_model.max_
    global_min = DE_model.global_min
    dir_path = DE_model.dir_path

    test_function_select = DE_model.test_function_selection
    test_function_evaluation = DE_model.test_function_evaluation

    mutation_type = DE_model.mutation_type

    # clustering parameters

    clustering_type = DE_model.clustering_type
    num_of_clusters = DE_model.num_of_clusters
    cluster_gen_begin = DE_model.cluster_gen_begin
    num_replace = DE_model.num_replace

    for i in np.arange(0,g):

        if i == 0:
            x = generate_initial_population(d, NP, test_function_name, min_, max_)
            gen_best = np.full((d, 1), 5)
        else:
            x = xgp

        # scaling factor F in [0,1] for difference vector
        
        NP_indices = list(np.arange(0,NP))   
        
        # mutation

        # DE/rand/1

        if mutation_type == 'random':
            y = mutate(d, NP, NP_indices, F, x)
        
        # DE/best/1

        if mutation_type == 'best':
            y = mutate_best(d, NP, NP_indices, F, gen_best, x)

        # constrain domain boundaries

        y = constrain_boundary(y, test_function_name)

        # crossover
        # crossover rate CR in[0,1] for each component trial vector acceptance

        z = crossover(d, NP, NP_indices, y, x, CR)

        # constrain domain boundaries

        z = constrain_boundary(z, test_function_name)

        # selection 

        xgp, z = selection(d, NP_indices, x, z, test_function_select)

        if False:
            svd_list, cond_num = plot_svd(xgp, i, dir_path, 'Generation' + str(i), svd_list)
            cond_num_values.append(cond_num)

        if d == NP:
            spectral = plot_evd(xgp, i, dir_path, 'Generation' + str(i), evd_list)
            spectral_values.append(spectral)
            ratio = spectral/spectral_values[0]
            print(f'spectral ratio = {ratio}')

        # evaluation of candidate solutions xgp vector wise
        # to determine if exit criteria has been met

        gen_function_value = test_function_evaluation(xgp, d)

        min_value = np.amin(gen_function_value)
        values.append(min_value)
        #result = np.where(gen_function_value == np.amin(gen_function_value))
        result = np.where(gen_function_value == np.amin(min_value))
        result = result[0][0] # index integer
        rank = np.linalg.matrix_rank(xgp, tol=1e-6)
        rank_values.append(rank)
        
        # clustering find centers

        #plot_gen(xgp,i,NP)
        
        if test_function_name == 'beale' and min_value <= global_min:
            break

        if test_function_name in ['eggholder', 'styb_tang', 'rosenbrock']:
            gb = xgp[:,result]
            num = global_min - gb
            #num = global_min - gb.reshape(d,1) # styb tang?
            relative_error = np.linalg.norm(num) / np.linalg.norm(global_min)
            #print(f'relative error = {relative_error}, i = {i}, F = {F}')
            relative_error_values.append(relative_error)
            if relative_error <= tol:
                print(f'Reached exit criteria at generation {i}')
                break
            # track improvement?
            residual = relative_error_values[i]-relative_error_values[i-1]
            #print(f'gen {i}, relative error {relative_error}, residual {residual}')
            if residual >= 0 and i >= cluster_gen_begin and clustering_type is not None: # == 0
                #print('clustering')
                resultm, centers, center_result = cluster_generation(i, d, xgp, gen_function_value, global_min, test_function_evaluation,
                                                                     clustering_type, num_of_clusters, num_replace)
                new_points = centers[center_result,:].T
                #xgp[:,resultm] = centers[center_result,:]
                xgp[:,resultm] = new_points
        #print(i)
        if i == 1500:
            breakpoint
        if not gb.shape == (d,1):
            breakpoint
        gen_best = np.full((d, 1), gb.reshape(d,1))
    optimum_point = xgp[:,result]
    #print(f'min test function value = {min_value}, index {result}, gen {i}')
    print(f'minimum point = {optimum_point}, index {result}, gen {i}, ending relative error = {relative_error}')
    #print(f'ending relative error = {relative_error}')
    return optimum_point, i, values, cond_num_values, spectral_values, relative_error_values, rank_values, xgp, svd_list, evd_list

# begin

# dir_path = os.path.dirname(os.path.realpath(__file__))

# d = 2 # solution space dimension
# NP = 40 # number of parameter vectors in each generation
# test_function_name = 'eggholder'

# initalization
# use boundary of parameter space to generate initial populatioin

# g = 200 # max number of generations
# values = []
# cond_num_values = []
# spectral_values = []
# relative_error_values = []
# rank_values = []
# global_min = return_vtr(test_function_name, d) # value to reach, depends on test function
# tol = 1e-5
# min_, max_ = create_extrema_dict(d, test_function_name)
# F = 0.95 # mutation scaling factor
# CR = 0.95 # crossover rate
# svd_list = np.empty( (d,1) )
# evd_list = np.empty( (d,1) )
# gen_best = np.full((d, 1), 1) 

# optimum_point, i, values, cond_num_values, spectral_values, relative_error_values, rank_values, final_gen, svd_list, evd_list = \
#     differential_evolution(g, test_function_name, values, cond_num_values, spectral_values, relative_error_values,
#                            rank_values, global_min, tol, min_, max_, F, CR, svd_list, evd_list, gen_best, eggholder)

#U, S, V_T = svd(xgp)

# plot function and singular values versus generation

# x = np.arange(0,i+1)
# y1 = values
# y2 = svd_list[0:][0] # change [0] to [1] for second largest svd_list[0:][1]
# y2 = y2[1:]

#plot_re_sv_gen(x, np.log(relative_error_values), svd_list)

# if d == NP:
#      plot_fv_spectral_gen(x, values, spectral_values)

# plot last generation points and global minimum

if False:

    plt.plot(final_gen[0,:,], final_gen[1,:], '.', markersize=2, label='GenPoint')
    plt.plot(global_min[0], global_min[1], '.', markersize=2, label='GlobalMinimum')
    plt.xlim(-550,550)
    plt.ylim(-550,550)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Final Generation Point' + str(NP))
    plt.legend(fontsize = 5)
    kfc = ('Final Generation Point', str(NP))
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep + 'images', output_name + '.png')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    #plt.show()
    plt.close()

# plot relative error versus generation

if False:

    plt.figure(figsize=(10,6))
    plt.plot(x, np.log(relative_error_values), label='Relative Error' )
    plt.xlabel('generation')
    plt.ylabel('log relative error')
    plt.title('Relative Error versus Generation')
    plt.legend()
    #plt.show()
    # save the plot as a file
    plt.savefig('images/relative-error-generation.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.close()

# svd

if False:
    fig, ax = plt.subplots(1, 1, figsize=(12,8) )
    pos = ax.imshow(svd_list)
    fig.colorbar(pos, ax=ax)
    plt.show()
