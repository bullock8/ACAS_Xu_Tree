from sklearn.linear_model import RidgeClassifier
from lineartree import LinearTreeClassifier
from sklearn.datasets import make_classification

from sklearn import tree
from sklearn.tree import _tree

X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = LinearTreeClassifier(base_estimator=RidgeClassifier())
clf.fit(X, y)

print(clf.summary())

