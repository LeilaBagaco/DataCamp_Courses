# ********** One-vs-rest SVM **********
# As motivation for the next and final chapter on support vector machines, we'll repeat the previous exercise with a non-linear SVM. 
# Once again, the data is loaded into X_train, y_train, X_test, and y_test .

# Instead of using LinearSVC, we'll now use scikit-learn's SVC object, which is a non-linear "kernel" SVM (much more on what this means in Chapter 4!). 
# Again, your task is to create a plot of the binary classifier for class 1 vs. rest.


# ********** Exercise Instructions **********
# 1 - Fit an SVC called svm_class_1 to predict class 1 vs. other classes.

# 2 - Plot this classifier.


# ********** Script **********

# We'll use SVC instead of LinearSVC from now on
from sklearn.svm import SVC

# Create/plot the binary classifier (class 1 vs. rest)
svm_class_1 = SVC()
svm_class_1.fit(X_train, y_train==1)
plot_classifier(X_train, y_train==1, svm_class_1)
