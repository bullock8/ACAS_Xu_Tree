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
import matplotlib.animation as animation

import onnxruntime as ort
from numba import njit

# decision tree stuff
from sklearn import tree
from sklearn.tree import _tree
from tqdm import tqdm
import pickle

from archive.acasxu_dubins_decision_tree import *

#######
# Load networks
#######
nets = load_networks()
net = nets[0]

#######
# Load Trees
#######
best_tree = pickle.load(open('best_tree.pickle', 'rb'))


test_pts = 10000
np.random.seed(seed = 23)
num_slices = 100

test_states = np.random.rand(num_slices, test_pts, 5)
tree_cmds = np.zeros([num_slices, test_pts])
net_cmds = np.zeros([num_slices, test_pts])
accuracy_scores = np.zeros([num_slices])

slice_min = -1/36 * np.pi
slice_max = 1/36 * np.pi
slice_var = np.linspace(-np.pi/36, np.pi/36, num = num_slices)
print(slice_var)

for slice_ind in range(num_slices):
    for i in range(test_pts):
        # Load the correct random state and scale it appropriately
        test_state = test_states[slice_ind, i, :]

        test_state = np.multiply(test_state, np.array([60760, 2 * np.pi, 2 * np.pi, 1100, 1200])) + np.array([0, -np.pi, -np.pi, 100, 0])
        
        # test_state = test_state.reshape(1, -1)
        #print(test_state.shape)

        # Fix 3 of the states, so we can compare
        test_state[0] = 60760/2
        test_state[1] = slice_var[slice_ind]#np.pi / 4
        test_state[2] = 0 #-np.pi / 4
        
        # Store the correctly scaled test state in the array
        test_states[slice_ind, i, :] = np.copy(test_state)

        # Save the ground-truth command from the neural net
        test_res = run_network(net, test_state)
        net_cmds[slice_ind, i] = np.argmin(test_res)

        # Save the predicted command from the decision tree
        #tree_cmd = best_tree.predict(test_state.reshape(1, -1))
        #tree_cmds[i] = tree_cmd.item()

    
    ######
    # Train Tree
    tree_cmds[slice_ind,:] = best_tree.predict(test_states[slice_ind, :, :])
    accuracy_scores[slice_ind] = best_tree.score(test_states[slice_ind, :, :], net_cmds[slice_ind, :])
    
print(f"accuracy over all the slices: {np.average(accuracy_scores)}")

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



#test_score = best_tree.score(test_states, net_cmds)
#print(f"Testing score:  {test_score}\n")
#get_cmap(net_cmds)
from matplotlib.colors import ListedColormap

#plt.figure(0)
#cmap, labels = get_cmap(net_cmds) #cm.rainbow(net_cmds / np.mean(net_cmds))

#coc_labels = np.where(labels == 'CoC')
#wl_labels = np.where(labels == 'WL')

values = ['CoC', 'WL', 'WR', 'SL', 'SR']

#plt.scatter(test_states[coc_labels, 3], test_states[coc_labels, 4], color = cmap[coc_labels], label = 'CoC')
#plt.scatter(test_states[wl_labels, 3], test_states[wl_labels, 4], color = cmap[wl_labels], label = 'WL')
colors = ['r', 'y', 'g', 'b', 'm']

def update_fig_gt(slice_ind):
    plt.clf()
    for i in range(0, len(values)):
        ix = np.where(net_cmds[slice_ind, :] == i)
        plt.scatter(test_states[slice_ind, ix, 3], test_states[slice_ind, ix, 4], c = colors[i], label = values[i]) 
    #plt.scatter(test_states[:, 3], test_states[:, 4], cmap = colors, c = net_cmds)
    pi_frac = slice_min + (slice_max - slice_min)*(slice_ind / num_slices)
    plt.title(f"Ground Truth (\\Theta = {pi_frac / np.pi} * pi)")
    plt.xlabel("v own")
    plt.ylabel("v int")
    plt.legend(loc='upper left')
    
ani = animation.FuncAnimation(fig = plt.figure(0), func = update_fig_gt, frames = num_slices, interval = 1000/5)

    
ani.save('groundTruth_anim_short.gif', writer = 'imagemagick', fps = 5)    





def update_fig_tree(slice_ind):
    plt.clf()
    for i in range(0, len(values)):
        ix = np.where(tree_cmds[slice_ind, :] == i)
        plt.scatter(test_states[slice_ind, ix, 3], test_states[slice_ind, ix, 4], c = colors[i], label = values[i]) 
    #plt.scatter(test_states[:, 3], test_states[:, 4], cmap = colors, c = net_cmds)
    pi_frac = slice_min + (slice_max - slice_min)*(slice_ind / num_slices)
    plt.title(f"Decision Tree (\\Theta = {pi_frac / np.pi} * pi)")
    plt.xlabel("v own")
    plt.ylabel("v int")
    plt.legend(loc='upper left')

ani2 = animation.FuncAnimation(fig = plt.figure(1), func = update_fig_tree, frames = num_slices, interval = 1000/5)

    
ani2.save('decisionTree_anim_short.gif', writer = 'imagemagick', fps = 5)    
    
#plt.show()


#plt.legend(['r', 'y', 'g', 'b', 'm'], ['CoC', 'WL', 'WR', 'SL', 'SR'])
'''
plt.figure(1)
#cmap2, labels2 = get_cmap(tree_cmds) #cm.rainbow(tree_cmds / np.mean(tree_cmds))
#plt.scatter(test_states[:, 3], test_states[:, 4], cmap = colors, c = tree_cmds)
for i in range(0, len(values)):
    ix = np.where(tree_cmds == i)
    plt.scatter(test_states[ix, 3], test_states[ix, 4], c = colors[i], label = values[i]) 
#plt.scatter(test_states[:, 3], test_states[:, 4], cmap = colors, c = net_cmds)
plt.legend()


plt.title("Decision Tree")
plt.xlabel("v own")
plt.ylabel("v int")
#plt.legend(*scatter.legend_elements())
#plt.legend()

fig = plt.figure(2)
ax = fig.add_subplot(projection = '3d')

for i in range(0, len(values)):
    ix = np.where(net_cmds == i)
    ax.scatter(test_states[ix, 3], test_states[ix, 4], test_states[ix,0], c = colors[i], label = values[i]) 
#plt.scatter(test_states[:, 3], test_states[:, 4], cmap = colors, c = net_cmds)
plt.legend()


ax.set_title("Ground Truth")
ax.set_xlabel("v own")
ax.set_ylabel("v int")
ax.set_zlabel("rho")


fig = plt.figure(3)
ax = fig.add_subplot(projection = '3d')

for i in range(0, len(values)):
    ix = np.where(tree_cmds == i)
    ax.scatter(test_states[ix, 3], test_states[ix, 4], test_states[ix,0], c = colors[i], label = values[i]) 
#plt.scatter(test_states[:, 3], test_states[:, 4], cmap = colors, c = net_cmds)
plt.legend()


ax.set_title("Decision Tree")
ax.set_xlabel("v own")
ax.set_ylabel("v int")
ax.set_zlabel("rho")
#plt.legend(*scatter.legend_elements())
#plt.legend()

plt.show()

'''

'''
#### Distance metric

# check through all test points
min_rho_dist = np.zeros([test_pts])
min_theta_dist = np.zeros([test_pts])
min_phi_dist = np.zeros([test_pts])
min_vown_dist = np.zeros([test_pts])
min_vint_dist = np.zeros([test_pts])


for i in range(test_pts):
    if net_cmds[i] != tree_cmds[i]:
        # find distance to nearest correctly-classified point
        incorrect_class = tree_cmds[i]
        ix = np.where(net_cmds == incorrect_class)
        min_rho_dist[i] = np.min(np.abs( test_states[i, 0] - test_states[ix, 0] ))
        min_theta_dist[i] = np.min(np.abs( test_states[i, 1] - test_states[ix, 1] ))
        min_phi_dist[i] = np.min(np.abs( test_states[i, 2] - test_states[ix, 2] ))
        min_vown_dist[i] = np.min(np.abs( test_states[i, 3] - test_states[ix, 3] ))
        min_vint_dist[i] = np.min(np.abs( test_states[i, 4] - test_states[ix, 4] ))

print(np.max(min_vint_dist))
'''


