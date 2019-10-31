
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# Load Iris sample dataset
iris = load_iris()
test_idx = [50,50,100]

# Training data
train_label =  np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Classifier
classifier = tree.DecisionTreeClassifier()
classifier.fit(train_data, train_label)

print(test_target)
print(classifier.predict(test_data))

print(iris.feature_names, iris.target_names)

# Visualization the tree
import graphviz
dot_data = tree.export_graphviz(classifier, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")



