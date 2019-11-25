# import dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Create the train model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

print("===================== X Train", X_train)
print("===================== Y Train", y_train)
print("===================== X Test", X_test)
print("===================== Y Test", y_test)
