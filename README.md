# ACAS Xu Tree Conversion
This work builds off [Stanley Bak's implementation of ACAS Xu](https://github.com/stanleybak/acasxu_closed_loop_sim/tree/main).  This repo contains The original 

## Installation
All required packages can be installed with the command

```
pip3 install -r requirements.txt
```

## Running Original ACAS Xu

To run the original ACAS Xu network simulation, as developed by Stanley Bak, run the `run_true_acas.py` script.

## Convert ACAS Xu to Decision Tree

To create the decision tree representation of the ACAS Xu neural networks, run the `create_decision_tree.py` script.

## Running DT ACAS

To run simulations which use the decision tree approximation of ACAS Xu, run the `run_DT_acas.py` script.

**Note:** this script loads the decision trees from pickle files in `saved_trees/`.  You will need to run `create_decision_tree.py` script at least once to create the necessary trees for `run_DT_acas.py`.

## Plotting Utilities

TODO:  Keep tree_compare.  tree_animate probably should be archived