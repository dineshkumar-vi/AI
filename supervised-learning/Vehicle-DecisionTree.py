from sklearn import tree

# Collect the training data
features =[[300, 2], [450, 2],[200, 8], [150, 9]]
labels = ["sport-car", "super-sport-car", "minivan", "SUV"]

# Classifier
classifier = tree.DecisionTreeClassifier()

# Fit the training data
classifier = classifier.fit(features, labels)

print("Output 1:" , classifier.predict([[200,8]]))
print("Output 2:" , classifier.predict([[800,1]]))

