from functools import lru_cache
import time
import math
import argparse

import numpy as np
from scipy import ndimage
from scipy.linalg import expm

import matplotlib.pyplot as plt
from matplotlib import patches, animation
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D
import matplotlib.cm as cm

import onnxruntime as ort
from numba import njit

# decision tree stuff
from sklearn import tree
from sklearn.tree import _tree
from tqdm import tqdm
import pickle
import os

from ACAS_Xu_networks.acas_xu_nets import *

#######
# Load networks
#######

def compare_net(net_ind, fixed_states):
    '''
    input:
        net_ind:  the index [0,4] of the network to compare with the DT
        fixed_states: a list of the 3 values to be fixed in the sample data (need 2D variation for plotting)
            fixed_states[0]:  fixed rho value
            fixed_states[1]:  fixed v_own value
            fixed_states[2]:  fixed v_int value
    output:
    '''
    nets = load_networks()
    net = nets[net_ind]

    #######
    # Load Trees
    #######
    best_tree = pickle.load(open(f'saved_trees{os.sep}tree_{net_ind}.pickle', 'rb'))


    test_pts = 10000
    np.random.seed(seed = 23)
    test_states = np.random.rand(test_pts, 5)
    tree_cmds = np.zeros([test_pts])
    net_cmds = np.zeros([test_pts])

    for i in range(test_pts):
        # Load the correct random state and scale it appropriately
        test_state = test_states[i]

        test_state = np.multiply(test_state, np.array([60760, 2 * np.pi, 2 * np.pi, 1100, 1200])) + np.array([0, -np.pi, -np.pi, 100, 0])
        
        # test_state = test_state.reshape(1, -1)
        #print(test_state.shape)

        # Fix 3 of the states, so we can compare
        test_state[0] = fixed_states[0]
        test_state[3] = fixed_states[1]
        test_state[4] = fixed_states[2]
        
        # Store the correctly scaled test state in the array
        test_states[i] = np.copy(test_state)

        # Save the ground-truth command from the neural net
        test_res = run_network(net, test_state)
        net_cmds[i] = np.argmin(test_res)

        # Save the predicted command from the decision tree
        #tree_cmd = best_tree.predict(test_state.reshape(1, -1))
        #tree_cmds[i] = tree_cmd.item()

        

    tree_cmds = best_tree.predict(test_states)


    def get_cmap(cmds):
        cmap_array = []
        labels = []
        for i in range(cmds.size):
            if cmds[i] == 0:
                cmap_array.append('r')
                labels.append('CoC')
            elif cmds[i] == 1:
                cmap_array.append('y')
                labels.append('WL')
            elif cmds[i] == 2:
                cmap_array.append('g')
                labels.append('WR')
            elif cmds[i] == 3:
                cmap_array.append('b')
                labels.append('SL')
            else:
                cmap_array.append('m')
                labels.append('SR')
        
        return cmap_array, labels



    test_score = best_tree.score(test_states, net_cmds)
    print(f"Testing score:  {test_score}\n")
    #get_cmap(net_cmds)
    from matplotlib.colors import ListedColormap

    plt.figure(0)
    cmap, labels = get_cmap(net_cmds) #cm.rainbow(net_cmds / np.mean(net_cmds))

    coc_labels = np.where(labels == 'CoC')
    wl_labels = np.where(labels == 'WL')

    values = ['CoC', 'WL', 'WR', 'SL', 'SR']

    #plt.scatter(test_states[coc_labels, 3], test_states[coc_labels, 4], color = cmap[coc_labels], label = 'CoC')
    #plt.scatter(test_states[wl_labels, 3], test_states[wl_labels, 4], color = cmap[wl_labels], label = 'WL')
    colors = ['r', 'y', 'g', 'b', 'm']

    for i in range(0, len(values)):
        ix = np.where(net_cmds == i)
        plt.scatter(test_states[ix, 1], test_states[ix, 2], c = colors[i], label = values[i]) 
    #plt.scatter(test_states[:, 3], test_states[:, 4], cmap = colors, c = net_cmds)
    plt.title("ACAS Xu NN")
    #ax.set_xlabel("v own")
    #ax.set_ylabel("v int")
    plt.xlabel(r"$\theta$ (rad)")
    plt.ylabel(r"$\psi$ (rad)")
    plt.legend()
    plt.savefig(f'saved_figs{os.sep}ACAS_True_{net_ind}.eps', format='eps')
    plt.savefig(f'saved_figs{os.sep}ACAS_True_{net_ind}.png', format='png')
    #plt.legend(['r', 'y', 'g', 'b', 'm'], ['CoC', 'WL', 'WR', 'SL', 'SR'])

    plt.figure(1)
    #cmap2, labels2 = get_cmap(tree_cmds) #cm.rainbow(tree_cmds / np.mean(tree_cmds))
    #plt.scatter(test_states[:, 3], test_states[:, 4], cmap = colors, c = tree_cmds)
    for i in range(0, len(values)):
        ix = np.where(tree_cmds == i)
        plt.scatter(test_states[ix, 1], test_states[ix, 2], c = colors[i], label = values[i]) 
    #plt.scatter(test_states[:, 3], test_states[:, 4], cmap = colors, c = net_cmds)
    plt.legend()


    plt.title("ACAS Xu Decision Tree")
    #ax.set_xlabel("v own")
    #ax.set_ylabel("v int")
    plt.xlabel(r"$\theta$ (rad)")
    plt.ylabel(r"$\psi$ (rad)")
    #plt.legend(*scatter.legend_elements())
    #plt.legend()
    plt.savefig(f'saved_figs{os.sep}ACAS_Tree_{net_ind}.eps', format='eps')
    plt.savefig(f'saved_figs{os.sep}ACAS_Tree_{net_ind}.png', format='png')

    plt.show()

if __name__ == "__main__":
    net_ind = 0
    fixed_states = [1000,100,100]
    compare_net(net_ind = net_ind, fixed_states=fixed_states)

