import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import logging
from scipy.linalg import svd
from numpy import array
import itertools
from sklearn.cluster import KMeans
from datetime import datetime
random.seed(42)
import ray

from DE import differential_evolution, return_vtr, create_extrema_dict, eggholder, DEModelClass

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

@ray.remote
def process(DE_model, values, cond_num_values, spectral_values, relative_error_values, rank_values, svd_list, evd_list, eggholder):

    optimum_point, i, values, cond_num_values, spectral_values, relative_error_values, rank_values, final_gen, svd_list, evd_list = \
    differential_evolution(DE_model, values, cond_num_values, spectral_values, relative_error_values, rank_values, svd_list, evd_list, eggholder)
    if i < g-1:
        outcome = 1
    if i == g-1:
        outcome = 0
    df = pd.DataFrame({'Run':[k], 'FinalGeneration':[i], 'RelativeError':[np.format_float_scientific(relative_error_values[i], precision=2)], 'F':[F], 'CR':[CR], 
                    'G':[g], 'Tolerance':[np.format_float_scientific(tol, precision=2)], 'd':[d], 'NP':[NP], 
                    'mutation_type':[mutation_type], 'clustering_type':[clustering_type], 'num_of_clusters':[num_of_clusters], 
                    'cluster_gen_begin':[cluster_gen_begin], 'num_replace':[num_replace],  'function':[test_function_name],
                    'Outcome':[outcome]} )
    kfc = (test_function_name, str(d), str(NP), str(g), str(F), str(CR), mutation_type, str(clustering_type), str(num_of_clusters), str(cluster_gen_begin), str(num_replace) )
    key = '-'.join(kfc)
    df['key'] = key
    #result_list.append(df)
    relative_error_values = []
    return df

def plot_difference_vector():

    # plot difference vectors

    # Define the vectors

    a = np.array([2,6])
    b = np.array([4,4])
    base = np.array([10,3])
    diff = a-b
    test = base + diff

    d = np.array([6,5])
    e = np.array([8,9])
    base2 = np.array([1,3])
    diff2 = e-d

    plt.figure(figsize=(10,6))
    # Plot the first vector (v1) as an arrow
    plt.quiver(b[0], b[1], diff[0], diff[1], angles="xy", scale_units="xy", scale = 1, color="Red", width=2e-3)
    # Plot the second vector (v2) as an arrow
    plt.quiver(base[0], base[1], diff[0], diff[1], angles="xy", scale_units="xy", scale = 1, color="cornflowerblue", width=2e-3)

    # plot points

    plt.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale = 1, color="darkgreen", width=2e-3)
    plt.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale = 1, color="gray", width=2e-3)
    plt.quiver(0, 0, base[0], base[1], angles="xy", scale_units="xy", scale = 1, color="darkmagenta", width=2e-3)
    plt.quiver(0, 0, test[0], test[1], angles="xy", scale_units="xy", scale = 1, color="maroon", width=2e-3)

    plt.plot(a[0], a[1], '.', markersize=10, label='a')
    plt.plot(b[0], b[1], '.', markersize=10, label='b')
    plt.plot(base[0], base[1], '.', markersize=10, label='r1')
    plt.plot(test[0], test[1], '.', markersize=10, label='test')
    #plt.plot(d[0], d[1], '.', markersize=10, label='d')
    #plt.plot(e[0], e[1], '.', markersize=10, label='e')

    # Set the limits for the plot
    plt.xlim([0,11])
    plt.ylim([0,7])

    # annotate
    plt.annotate('r2', # this is the text
                    (a[0],a[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    fontsize=13,
                    ha='center') # horizontal alignment can be left, right or center

    plt.annotate('r3', # this is the text
                    (b[0],b[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    fontsize=13,
                    ha='center') # horizontal alignment can be left, right or center

    plt.annotate('r1', # this is the text
                    (base[0],base[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,15), # distance from text to points (x,y)
                    fontsize=13,
                    ha='center') # horizontal alignment can be left, right or center

    plt.annotate('r2-r3', # this is the text
                    (b[0],b[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(-35,50), # distance from text to points (x,y)
                    fontsize=13,
                    ha='center') # horizontal alignment can be left, right or center

    plt.annotate('r2-r3', # this is the text
                    (base[0],base[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(-35,50), # distance from text to points (x,y)
                    fontsize=13,
                    ha='center') # horizontal alignment can be left, right or center

    plt.annotate('r', # this is the text
                    (test[0],test[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(-10,10), # distance from text to points (x,y)
                    fontsize=13,
                    ha='center') # horizontal alignment can be left, right or center

    # Set the labels for the plot
    plt.xlabel('x')
    plt.ylabel('y')

    # Show the grid lines
    plt.grid()
    #plt.show()

    output_name = 'difference_vector'
    output_loc = os.path.join(dir_path + os.sep + 'images', output_name + '.png')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    plt.close()


# begin

start = datetime.now()

dir_path = os.path.dirname(os.path.realpath(__file__))

# parameter exploration
# DE parameters

DE_grid = {'d':[2],
           'NP':[100], 
           'g':[200], #
           'F':[0.95], 
           'CR': [0.95], 
           'test_function_name': ['eggholder'], # rosenbrock, eggholder
           'tol': [1e-5], 
           'mutation_type': ['best'], 
           'clustering_type': ['kmeans'], # kmeans, spectral, agg
           'num_of_clusters': [50], # 50
           'cluster_gen_begin': [10], # 10
           'num_replace': [10,15,20], # 9
}

a = DE_grid.values()
combinations = list(itertools.product(*a))

result_list = []
master = []

#ray.init(num_cpus=8)
for param in combinations:
        ray.init(num_cpus=8)
        print(f'Starting parallel DE exploration {param}')
        d = param[0] # dimension
        NP = param[1] # number of parameter vectors in each generation 
        g = param[2] # max number of generations
        F = param[3] # mutate scaling factor
        CR = param[4] # crossover rate 
        test_function_name = param[5]
        tol = param[6]
        mutation_type = param[7]
        clustering_type = param[8]
        num_of_clusters = param[9]
        cluster_gen_begin = param[10]
        num_replace = param[11]

        # initialize class

        DE_model = DEModelClass(d, NP, g, F, CR, test_function_name, tol,
                        mutation_type, clustering_type, num_of_clusters, cluster_gen_begin, num_replace)

        values = []
        cond_num_values = []
        spectral_values = []
        relative_error_values = []
        rank_values = []
        svd_list = np.empty( (d,1) )
        evd_list = np.empty( (d,1) )

        # run DE for steps

        runs = 1000
        for k in np.arange(0, runs):
            logging.info(f'starting run {k}')
            result_list.append(process.remote(DE_model, values, cond_num_values, spectral_values, relative_error_values, rank_values, svd_list, evd_list, eggholder ) )

        results = ray.get(result_list)
        results = pd.concat(results, sort=False)
        master.append(results)
        #results = []
        result_list = []
        ray.shutdown()

data = pd.concat(master, sort=False)
#ray.shutdown()

summary = data.groupby(['key'])['Outcome'].sum()
summary = summary.reset_index(drop = False)
summary = pd.DataFrame(summary, columns = ['key', 'Outcome'])
summary = summary.rename(columns = {'Outcome':'OutcomeSum'})
print(summary)
time_taken = datetime.now() - start
logging.info(f'runtime was {time_taken}')

# output

kfc = ('exploration', test_function_name, str(DE_model.clustering_type), str(runs)) 
out_dir = r'/home/wesley/repos/rbf-fd/data'
output_name = '-'.join(kfc)
output_loc = os.path.join(out_dir + os.sep + output_name + '.ods')

logging.info(f'Saving to {output_loc}')
with pd.ExcelWriter(output_loc ) as writer:
    data.to_excel(writer, sheet_name = 'data', index=False)
    summary.to_excel(writer, sheet_name = 'summary', index=False)


# plot parameter exploration

cols = ['function', 'd', 'NP', 'g', 'F', 'CR', 'mutation_type', 'clustering_type', 'num_of_clusters', 'cluster_gen_begin', 'num_replace']
summary[cols] = summary.key.str.split("-", expand=True)

xcol = 'num_replace'
ycol = 'OutcomeSum'
name1 = 'eggholder10d'
name2 = 'spectral'

x = summary[xcol]
y = summary[ycol]

# plt.figure(figsize=(6,6))
# #plt.xlim(-550,550)
# #plt.ylim(-550,550)
# plt.plot(x, y, label='kmeans')
# plt.xlabel('Maximum Generation')
# plt.ylabel('Total Success')
# #plt.title('k-means cluster')
# plt.legend(fontsize = 8, loc='center right')
# plt.savefig(f'images/{name1}-{name2}-exploration-{xcol}-vs-{ycol}.png',
#             format='png',
#             dpi=300,
#             bbox_inches='tight')
# plt.show()
# a = True
# #plt.close()


