# decision tree imports
from sklearn import tree
from sklearn.tree import _tree
from tqdm import tqdm
import pickle
import os

from ACAS_Xu_networks.acas_xu_nets import *

import warnings
warnings.filterwarnings('ignore')

# This function reads the tree object and prints out a block of 
# Python "if" statements.  You can also pipe the output to its
# own text file (what I did)
def tree_to_code(tree, feature_names, net_num, file_handle):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
    print(f"def predict_{net_num}({", ".join(feature_names)}):", file = file_handle)

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, np.round(threshold,2)), file = file_handle)
            recurse(tree_.children_left[node], depth + 1)
            #print("{}else:  # if {} > {}".format(indent, name, np.round(threshold,2)))
            print("{}if {} > {}:".format(indent, name, np.round(threshold,2)), file = file_handle)
            recurse(tree_.children_right[node], depth + 1)
        else:
            next_cmd = np.argmax(tree_.value[node])
            
            if next_cmd==0:
                mode_str = 'CraftMode.Coc'
            if next_cmd==1:
                mode_str = 'CraftMode.Weak_left'
            if next_cmd==2:
                mode_str = 'CraftMode.Weak_right'
            if next_cmd==3:
                mode_str = 'CraftMode.Strong_left'
            if next_cmd==4:
                mode_str = 'CraftMode.Strong_right'
            
            print("{}next.agent_mode = {}".format(indent, mode_str), file = file_handle)
            print("{}next.timer = 0".format(indent), file = file_handle)

    recurse(0, 1)
    
    
def create_single_tree(net_index, training_size, testing_size, tree_alpha, save_tree = True):
    '''
    input:
        (int) net_index: index of the ACAS neural net that you want to approximate (from 0 to 4, inclusive)
        (list) training_size: a five-element list of integers.  Each integer corresponds to number of desired training points for each relative ACAS Xu state
                training_size[0]: number of rho samples
                training_size[1]: number of theta samples
                training_size[2]: number of psi samples
                training_size[3]: number of v_own samples
                training_size[4]: number of v_int samples               
        (int) testing_size: Number of test points to generate for testing the tree accuracy (integer)
        (double)tree_alpha: this specifies the value of the alpha parameter for pruning the tree
        (bool) save_tree:  Boolean for saving the trained DecisionTreeClassifier in a .pickle file (True = save tree, False = don't save)
    output:
        tree_{net_index}.pickle: a trained DecisionTreeClassifier for the ACAS Xu network at net_index
        tree_{net_index}.txt:  a text file of the Python if-statements inside the trained decision tree
    '''
    #############################################################
    ############## Load ACAS Xu Networks ########################
    #############################################################
    nets = load_networks()
    
    # Choose the specific network you want to use here
    net = nets[net_index]


    ############################
    ###### Handy ACAS Info #####
    ############################
    
    # 0: rho, distance
    # 1: theta, angle to intruder relative to ownship heading
    # 2: psi, heading of intruder relative to ownship heading
    # 3: v_own, speed of ownship
    # 4: v_int, speed in intruder

    # min inputs: 0, -3.1415, -3.1415, 100, 0
    # max inputs: 60760, 3.1415, 3,1415, 1200, 1200
    
    #############################################################
    ############## Training Data Generation #####################
    #############################################################    

    # Set number of samples for each variable's training data
    num_rhos = training_size[0] 
    num_thetas = training_size[1]
    num_psis = training_size[2]
    num_vOwns = training_size[3]
    num_vInts = training_size[4]
    
    # Generate your sample arrays for each ACAS state
    rho_range = np.linspace(0, 60760, num_rhos)
    theta_range = np.linspace(-np.pi, np.pi, num_thetas)
    psi_range = np.linspace(-np.pi, np.pi, num_psis)
    v_own_range = np.linspace(100, 1200, num_vOwns)
    v_int_range = np.linspace(0, 1200, num_vInts)       
        

    # stored_states will store all possible combinations of your sampled ACAS training states
    stored_states = np.zeros([num_rhos * num_thetas * num_psis * num_vOwns * num_vInts, 5])
    
    # command_nums stores the corresponding ACAS advisory {0,1,2,3,4} for each entry in stored_states
    command_nums = np.zeros([num_rhos * num_thetas * num_psis * num_vOwns * num_vInts])

    # I didn't want to do too much math to keep track of which array index I'm at, so I just created an index variable and increment it for each loop
    index = 0
    for rho_ind in range(0, num_rhos):
        for theta_ind in range(0, num_thetas):
            for psi_ind in range(0, num_psis):
                for v_own_ind in range(0, num_vOwns):
                    for v_int_ind in range(0, num_vInts):
                            
                        rho = rho_range[rho_ind]
                        theta = theta_range[theta_ind]
                        psi = psi_range[psi_ind]
                        v_own = v_own_range[v_own_ind]
                        v_int = v_int_range[v_int_ind]

                        # Relative state list for ACAS
                        state = [rho, theta, psi, v_own, v_int]
                        
                        # I think you have to copy() here, otherwise you have issues with run_network() overwriting the stored 
                        # states with the normalized stored states
                        stored_states[index, :] = state.copy()
                        
                        # Run ACAS Xu and get the resulting turn command
                        res = run_network(net, state)
                        command = np.argmin(res)

                        # Save the command and increment the index
                        command_nums[index] = command

                        index += 1
    
        
    #############################################################
    ############## Testing Data Generation ######################
    #############################################################   
    
    # Set number of test points to generate
    test_pts = testing_size
    # Set a seed for debugging purposes (I like 23)
    np.random.seed(seed = 23)
    
    # Generate uniform random numbers for test states
    test_states = np.random.rand(test_pts, 5)
    test_cmds = np.zeros([test_pts])
    
    # Scale the test points to the appropriate range for each variable
    for i in range(0, test_pts):
        test_state = test_states[i]
        
        # rescale and shift the test state
        test_state = np.multiply(test_state, np.array([60760, 2 * np.pi, 2 * np.pi, 1100, 1200])) + np.array([0, -np.pi, -np.pi, 100, 0])
        
        # Again, copy the test state so run_network() doesn't overwrite anything
        test_states[i] = test_state.copy()
        
        # Calculate and safe the true ACAS command for each test states
        test_res = run_network(net, test_state)
        test_cmd = np.argmin(test_res)
        
        test_cmds[i] = test_cmd
            
    #############################################################
    ############## Create Decision Tree #########################
    #############################################################  
    
    clf = tree.DecisionTreeClassifier()
    
    # Set alpha parameter for minimum cost-complexity pruning
    # https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning
    best_alpha = tree_alpha
    
    # Create decision tree and fit to the training data
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha, class_weight='balanced')
    clf.fit(stored_states, command_nums)
    print(
        "\tNumber of nodes in the tree is: {} with ccp_alpha: {}".format(
            clf.tree_.node_count, best_alpha
        )
    )

    # Compare training and testing accuracy of the pruned trees   
    train_score = clf.score(stored_states, command_nums)
    print(f"\tTraining score:  {train_score}")
    test_score = clf.score(test_states, test_cmds)
    print(f"\tTesting score:  {test_score}")

    # Save the tree object if you want to keep it for further plotting
    if save_tree:
        pickle.dump(clf, open(f'saved_trees{os.sep}tree_{net_index}.pickle', 'wb'))


    # This is nice if you want a text readout, but not formatted for Python "if" statements
    text_representation = tree.export_text(clf)
    #print(text_representation)

    # with open("decision_tree.txt", "w") as fout:
    #     fout.write(text_representation)
    
    # This prints out the decision tree represented by Python "if" statements
    # Handy if you have a large tree

    with open(f"saved_trees{os.sep}tree_{net_index}.txt", "a") as f:
        tree_to_code(clf, ['rho', 'theta', 'psi', 'vOwn', 'vInt'], net_index, file_handle = f)

if __name__ == "__main__":
    
    net_index = 0
    train_list = [30, 10, 10, 10, 10]
    test_size = 1000
    alpha = 0.002
    
    for net_ind in range(0, 5):
        print(f"Creating tree from network {net_ind}")
        create_single_tree(net_index=net_ind, training_size=train_list, testing_size=test_size, tree_alpha=alpha, save_tree=True)
        print("")


    
