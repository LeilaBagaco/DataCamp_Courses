# ********** Running LogisticRegression and SVC **********
# In this exercise, you'll apply logistic regression and a support vector machine to classify images of handwritten digits.

# ********** Exercise Instructions **********
# 1 - Apply logistic regression and SVM (using SVC()) to the handwritten digits data set using the provided train/validation split.

# 2 - For each classifier, print out the training and validation accuracy.

# ********** Script **********

from sklearn import datasets
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

# Apply logistic regression and print scores
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

# Apply SVM and print scores
svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))
