import pickle
import matplotlib.pyplot as plt

ccp_alphas = pickle.load(open('alphas.pickle', 'rb'))
train_scores = pickle.load(open('trainScores.pickle', 'rb'))
test_scores = pickle.load(open('testScores.pickle', 'rb'))
impurities = pickle.load(open('impurities.pickle', 'rb'))

# Plot impurity vs alpha value for the tree
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()