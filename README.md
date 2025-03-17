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

You can plot 2D representations of the decision tree and neural net command advisories, as seen below.  To create these figures, run `compare_ACAS_NN_DT.py`.
![Tree Representation](https://github.com/bullock8/ACAS_Xu_Tree/blob/a6606ddc9f011152b77b2686507c1f891d2cfce9/archive/ACAS_Tree.png) ![NN Representation](https://github.com/bullock8/ACAS_Xu_Tree/blob/main/archive/ACAS_True.png)

