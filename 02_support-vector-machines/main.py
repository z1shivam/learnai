from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()

X = data.data
Y = data.target

# if you want to have same state all the time, set random state to a fixed number
x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    #  random_state=20
)

clf = SVC(kernel="linear", C=3)
clf.fit(x_train, y_train)

clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(x_train, y_train)

print("k neighbours classifier -> ", clf2.score(x_test, y_test))
print("Support Vector Machines -> ", clf.score(x_test, y_test))